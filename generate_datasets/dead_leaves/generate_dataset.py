import sys, os
assert os.getcwd().endswith('generate_datasets'), "This script should be run from generate_datasets folder."

from p_tqdm import p_map

from utils_generation import *
from torchvision.datasets.folder import pil_loader

# ported from dead_leaves.m
def dead_leaves(res, sigma, shape_mode='mixed', max_iters=5000, textures=None):
    img = np.zeros((res, res, 3), dtype=np.uint8)
    rmin = 0.03
    rmax = 1

    # compute distribution of radiis (exponential distribution with lambda = sigma):
    k = 200
    r_list = np.linspace(rmin, rmax, k)
    r_dist = 1./(r_list ** sigma)
    if sigma > 0:
        # normalize so that the tail is 0 (p(r >= rmax)) = 0
        r_dist = r_dist - 1/rmax**sigma
    r_dist = np.cumsum(r_dist)
    # normalize so that cumsum is 1.
    r_dist = r_dist/r_dist.max()

    for i in range(max_iters):
        available_shapes = ['circle', 'square', 'oriented_square','rectangle', 'triangle', 'quadrilater']
        assert shape_mode in available_shapes or shape_mode == 'mixed'
        if shape_mode == 'mixed':
            shape = random.choice(available_shapes)
        else:
            shape = shape_mode

        color = tuple([int(k) for k in np.random.uniform(0, 1, 3) * 255])

        r_p = np.random.uniform(0,1)
        r_i = np.argmin(np.abs(r_dist - r_p))
        radius = max(int(r_list[r_i] * res), 1)

        center_x, center_y = np.array(np.random.uniform(0,res, size=2),dtype='uint8')
        if shape == 'circle':
            img = cv2.circle(img, (center_x, center_y),radius=radius, color=color, thickness=-1)
        else:
            if shape == 'square' or shape == 'oriented_square':
                side = radius * np.sqrt(2)
                corners = np.array(((- side / 2, - side / 2),
                                    (+ side / 2, - side / 2),
                                    (+ side / 2, + side / 2),
                                    (- side / 2, + side / 2)), dtype='int32')
                if shape == 'oriented_square':
                    theta = np.random.uniform(0, 2 * np.pi)
                    c, s = np.cos(theta), np.sin(theta)
                    R = np.array(((c, -s), (s, c)))
                    corners = (R @ corners.transpose()).transpose()
            elif shape == 'rectangle':
                # sample one points in the firrst quadrant, and get the two other symmetric
                a = np.random.uniform(0, 0.5*np.pi, 1)
                corners = np.array(((+ radius * np.cos(a), + radius * np.sin(a)),
                                    (+ radius * np.cos(a), - radius * np.sin(a)),
                                    (- radius * np.cos(a), - radius * np.sin(a)),
                                    (- radius * np.cos(a), + radius * np.sin(a))), dtype='int32')[:,:,0]

            else:
                # we sample three or 4 points on a circle of the given radius
                angles = sorted(np.random.uniform(0, 2*np.pi, 3 if shape == 'triangle' else 4))
                corners = []
                for a in angles:
                    corners.append((radius * np.cos(a), radius * np.sin(a)))

            corners = np.array((center_x, center_y)) + np.array(corners)
            img = cv2.fillPoly(img, np.array(corners, dtype='int32')[None], color=color)
        if (img.sum(-1) == 0).sum() == 0:
            break

    img = img.transpose((2,0,1))

    if not textures is None:
        # find unique values and for each put a texture
        different_colors = img[0] * 256 **2 + img[1] * 256 + img[2]
        for color in np.unique(different_colors):
            texture_f = random.choice(list(textures.keys()))
            texture_image = loaded_textures[texture_f]

            #assert texture_image.shape == img.shape, "Precomputed textures should have the same shape as the dataset to be computed"
            mask = different_colors == color
            img = img * (1 - mask) + texture_image * mask
    return np.clip(img, 0, 255)

def generate_dataset(shape_mode, output_path, resolution=96, parallel=True, N_samples=105000, textures=None):
    sigma = 3

    os.makedirs(output_path, exist_ok=True)

    def generate_single_train(i):
        folder_path = output_path + '/' + str((i // 1000) * 1000).zfill(10)
        os.makedirs(folder_path, exist_ok=True)
        image_path = folder_path + '/' + str(i).zfill(10) + '.jpg'
        error=False
        try:
            if os.path.exists(image_path):
                # open image, if it fails create again
                img = pil_loader(image_path)
                assert img.size == (resolution, resolution)
                return
        except:
            print("Error in an image {}, will regenerate!".format(i))
            error = True
            pass
        random.seed(i)
        np.random.seed(i)
        im = dead_leaves(resolution, sigma, shape_mode=shape_mode, textures=textures)
        if not os.path.exists(image_path) or error:
            # check again for race condition, except if there was a previous error
            cv2_imwrite(im, image_path)

    to_generate = list(range(N_samples))
    import time
    random.seed(time.time())
    random.shuffle(to_generate)

    # train
    if parallel:
        p_map(generate_single_train, to_generate)
    else:
        for i in tqdm(to_generate):
            generate_single_train(i)

def subdivide_folders(base_folder):
    all_files = sorted(listdir(base_folder, extension='.jpg'))
    for f in tqdm(all_files):
        id = int(f.split('/')[-1].replace('.jpg', ''))
        folder_file = base_folder + '/' + str((id // 1000) * 1000).zfill(10)
        os.makedirs(folder_file, exist_ok=True)
        shutil.move(base_folder + '/' + f, folder_file + '/' + f)

def large_scale_options(opt):
    opt.resolution = 256
    opt.samples = 1300000
    return opt

def small_scale_options(opt):
    opt.resolution = 128
    opt.samples = 105000
    return opt

def get_output_path(opt):
    if len(opt.output_folder) == 0:
        output_path = '../generated_datasets/{}/dead_leaves-'.format('small_scale' if opt.small_scale else 'large_scale')
        if not opt.small_scale or opt.large_scale:
            raise Exception("When not using --small-scale or --large-scale the outputp path should be defined!")
        # Use default names
        if not opt.textured:
            if opt.shape_model == 'mixed':
                output_path += 'mixed/train'
            elif opt.shape_model == 'oriented_square':
                output_path += 'oriented/train'
            elif opt.shape_model == 'square':
                output_path += 'squares/train'
            else:
                raise Exception("Not one of the default modes, so output_path has to be passed as an argument!")
        elif opt.shape_model == 'mixed' and opt.texture_folder.endswith('stat-spectrum_color_wmm/train'):
            output_path += 'textures/train'
        else:
            raise Exception("Not one of the default modes, so output_path has to be passed as an argument!")
    else:
        output_path = opt.output_folder
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Statistical image generation')

    parser.add_argument('--small-scale', action="store_true", help="use small-scale default parameters")
    parser.add_argument('--large-scale', action="store_true", help="use large-scale default parameters")

    parser.add_argument('--output-folder', default='', type=str, help='Output folder where to dump the datasets')
    parser.add_argument('--output-file-type', type=str, default='jpg', choices=['jpg', 'png'], help='Filetype to generate')

    parser.add_argument('--resolution', type=int, default=128, help='Resolution to use')
    parser.add_argument('--samples', type=int, default=105000, help='N samples to generate')

    parser.add_argument('--textured', action="store_true", help="use textures")
    parser.add_argument('--texture-folder', default='../generated_datasets/small_scale/stat-spectrum_color_wmm/train', help='texture folder to load. 10k samples must be available')

    parser.add_argument('--parallel', type=str2bool, default="False", help='Whether to apply a random rotation to the color channels so that they are correlated')
    parser.add_argument('--shape-model', type=str, default="square", choices=['square', 'oriented_square', 'mixed'], help='What type of shapes to use')

    opt = parser.parse_args()

    if opt.large_scale:
        opt = large_scale_options(opt)
    elif opt.small_scale:
        opt = small_scale_options(opt)
    output_folder = get_output_path(opt)

    if opt.textured:
        textures_wmm_folders = sorted(listdir(opt.texture_folder, prepend_folder=True))
        texture_files = []
        print("listing textures wmm")
        n_textures_to_load = 10000
        for dir in tqdm(textures_wmm_folders):
            texture_files.extend(listdir(dir, prepend_folder=True))
        print("End listing textures wmm")

        assert len(texture_files) >= n_textures_to_load
        texture_files = random.sample(texture_files, n_textures_to_load)
        loaded_textures = dict()

        print("Loading textures to memory!")
        for texture_f in tqdm(texture_files):
            loaded_textures[texture_f] = cv2_resize(cv2_imread(texture_f), (opt.resolution, opt.resolution))
        print("Ended textures to memory!")

    else:
        loaded_textures = None

    generate_dataset(opt.shape_model, output_folder, resolution=opt.resolution, textures=loaded_textures, N_samples=opt.samples, parallel=opt.parallel)