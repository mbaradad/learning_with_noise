import sys

from utils_generation import *
from p_tqdm import p_map

import pyrtools as pt
from torchvision.datasets.folder import pil_loader

from my_python_utils.common_utils import *

# to control steerable pyramid parameters for all calls to pt.pyramids.SteerablePyramidFreq()
def get_steerable_pyramid(img, Nscales):
    assert len(img.shape) == 2, "Image passed to has to have len(img.shape) == 2, and found {}".format(img.shape)

    return pt.pyramids.SteerablePyramidFreq(img, height=Nscales)

class GenerationParams():
    def __init__(self, resolution=128,
                 Niter=10,
                 impose_color_model=False,
                 correlate_channels=False,
                 impose_spectrum=False,
                 impose_wmm=False,
                 slope_range=[0.5,3.5],
                 debug=False,
                 base_image='',
                 delta_p_range_0=0,
                 delta_p_range_1=0.9,
                 **kwargs):

        self.slope_range = slope_range # range of slope values for the power spectrum
        self.Niter = Niter # Number of iterations used to set the image model parameters

        self.resolution = resolution
        self.debug = debug
        self.delta_p_range = [delta_p_range_0, delta_p_range_1]

        self.base_image = base_image

        self.impose_color_model = impose_color_model
        self.correlate_channels = correlate_channels
        self.impose_spectrum = impose_spectrum
        self.impose_wmm = impose_wmm

def sample_colormodel(p):
    resolution = p.resolution
    Ncolors = int(3 + np.floor(np.random.uniform(0,20)))
    P = 0.001 + np.random.uniform(0,1, Ncolors)
    P /= P.sum()
    counts = np.random.multinomial(resolution ** 2, P)

    k = 0
    index_image = np.zeros(resolution ** 2, dtype='uint8')
    for i, count in enumerate(counts):
        index_image[k:k+count] = i
        k += count
    index_image = index_image.reshape(resolution, resolution)

    colors = np.array(np.floor(np.random.uniform(0,256, size=(Ncolors, 3))), dtype='uint8')
    ref = colors[index_image].transpose((2,0,1))

    m = int(np.ceil(np.random.uniform(0,10)))

    from scipy.signal import convolve2d

    for c in range(3):
        ref[c] = convolve2d(ref[c], np.ones((m,m))/m/m, 'same', boundary='symm')
    ref = ref + np.random.uniform(0, 10)
    ref = ref - ref.min()
    ref = np.array(255 * ref/ref.max(), dtype='uint8')

    # TODO: in the old matlab scripts, we were returning ref instead of the img_decor, check if it affects
    img_decor, colorPC, MeanColor = colorPCA(ref)
    HistIma, HistImaBins, HistImaQuantiles = get_image_color_histograms(img_decor)

    return colorPC, MeanColor, HistIma, HistImaBins, HistImaQuantiles, img_decor

def colorPCA(ref_image):
    import scipy.linalg as sla

    c, h, w = ref_image.shape
    img = ref_image.reshape(3, h * w)
    MeanColor = img.mean(-1)

    img = img - MeanColor[:,None]

    X = img @ img.transpose()

    _ , colorPC = sla.eig(X)
    img = colorPC @ img
    img = img.reshape((3, h, w))

    return img, colorPC, MeanColor

def inv_colorPCA(input_image, p):
    colorPC = p.colorPC
    meanColor = p.MeanColor

    c, h, w = input_image.shape
    img = input_image.reshape(c, h * w)
    out = colorPC.transpose() @ img
    assert len(meanColor.shape) == 1
    out += meanColor[:,None]
    out = out.reshape((c,h,w))
    return out

def get_image_color_histograms(img_decor, nbins = 256):
    histImg = []
    histImgBins = []
    histImgQuantiles = []

    for i in range(3):
        hist, bin_edges = np.histogram(img_decor[i], bins=nbins)

        histImg.append(hist)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
        histImgBins.append(bin_centers)
        histImgQuantiles.append(np.cumsum(hist) / img_decor[i].size)

    return np.array(histImg), np.array(histImgBins), np.array(histImgQuantiles)

def impose_spectrum(tmp, p):
    for c in range(3):
        x = tmp[c]

        m = np.mean(x)
        s = np.std(x)

        fft_img = np.fft.fft2(x - m)
        fft_phase = np.angle(fft_img)

        fft_imposed = p.SpectrumMagnitude[c] * np.exp(1j * fft_phase)
        y = np.real(np.fft.ifft2(fft_imposed))

        y = y - y.mean()
        y = y / np.std(y) * s + m

        tmp[c] = y

    return tmp

def impose_wmm(img, p):
    tmp = np.array(img)
    for color in range(3):
        img_c = tmp[color]
        py_c = get_steerable_pyramid(img_c, p.Nscales)

        for k in p.HistbBins.keys():
            # get the pyramid keys corresponding to the actual channel, as we sample independently per each of the three channels
            if k[0] != color:
                continue
            x = py_c.pyr_coeffs[k[1]]
            x_before = np.array(x)

            x = (x - np.mean(x)) / np.std(x)
            x = x * p.stdb[k] + p.meanb[k]

            x = match_cumulative_cdf(x, p.HistQuantiles[k], p.HistbBins[k])

            x = (x - np.mean(x)) / np.std(x)
            x = x * p.stdb[k] + p.meanb[k]

            py_c.pyr_coeffs[k[1]] = x

        tmp[color] = np.array(py_c.recon_pyr())

    return tmp

def impose_color_model(tmp, p):
    for c in range(3):
        tmp[c] = match_cumulative_cdf(tmp[c], p.HistImaQuantiles[c], p.HistImaBins[c])
    return tmp

def match_cumulative_cdf(source, tmpl_quantiles, tmpl_values):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    src_values, src_unique_indices, src_counts = np.unique(source.ravel(),
                                                           return_inverse=True,
                                                           return_counts=True)

    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / source.size

    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    matched_data = interp_a_values[src_unique_indices].reshape(source.shape)
    return matched_data

def sample_image(input, p, reconstruct_colors):
    tmp = input

    for iter in range(p.Niter):
        if p.impose_spectrum:
            tmp = impose_spectrum(tmp, p)

        if p.impose_wmm:
            tmp = impose_wmm(tmp, p)

        if p.impose_color_model:
            tmp = impose_color_model(tmp, p)

    if reconstruct_colors:
        tmp = inv_colorPCA(tmp, p)

    # normalize again the image
    tmp = 255 * (tmp - tmp.min()) / tmp.max()

    return tmp

def sample_1f_spectrum(p):
    resolution = p.resolution

    slope_0, slope_1 = p.slope_range
    s = np.random.uniform(slope_0, slope_1)
    d = np.random.normal() * np.abs(slope_1 - slope_0) / 15 / 4

    slope_x = s + d
    slope_y = s - d

    fx, fy = np.meshgrid(range(resolution), range(resolution))

    fx = fx - resolution / 2
    fy = fy - resolution / 2

    fr = 1e-16 + np.abs(fx / resolution) ** slope_x + + np.abs(fy / resolution) ** slope_y

    magnitude = 1 / np.fft.fftshift(fr)
    magnitude[0,0] = 0
    magnitude = np.tile(magnitude[None,:,:], (3,1,1))

    return magnitude

def sample_wmm_model(p, delta_at_0=False):
    resolution = p.resolution

    test_canvas = np.random.uniform(0,1, size=(resolution, resolution))
    assert p.resolution in [128, 256], "Number of scales not defined for resolution {}".format(p.resolution)
    p.Nscales = 3 if p.resolution == 128 else 4
    py = get_steerable_pyramid(test_canvas, p.Nscales)

    x = np.mgrid[-200:200.5:0.5]
    # the original MATLAB script does not modify the lowpass-residual (it iterates over 1:Nscales*Nor+1 when constructing/imposing the , which does not include the last residual)
    # not sure if it is done on purpose or not
    sp_keys = [ k for k in py.pyr_size if not 'residual_lowpass' in k]
    sp_key_to_size = dict([(k, py.pyr_size[k][0]) for k in sp_keys])

    PyramidCoeffs = dict()
    HistbBins = dict()
    Histb = dict()
    HistQuantiles = dict()
    meanb = dict()
    stdb = dict()

    for c in range(3):
        for k, size in sp_key_to_size.items():
            scale = resolution / size
            shat = 4 ** scale
            rhat = 0.4 + 0.4 * np.random.uniform(0,1)
            h = np.exp(-np.abs(x/shat) ** rhat)

            if delta_at_0:
                mix_probability = np.random.uniform(p.delta_p_range[0], p.delta_p_range[1])
                pos_0 = np.argmin(np.abs(x))
                h_sum = h.sum()
                h = h * (1 - mix_probability)
                h[pos_0] = h_sum * mix_probability

            m = np.sum(x * h) / np.sum(h)
            s = np.sqrt(np.sum((x - m)**2 * h) / np.sum(h))

            actual_pyramid_key = (c,k)

            HistbBins[actual_pyramid_key] = x
            Histb[actual_pyramid_key] = h
            HistQuantiles[actual_pyramid_key] = np.cumsum(h) / h.sum()
            meanb[actual_pyramid_key] = m
            stdb[actual_pyramid_key] = s

    # pyramid coeffs are not set, but returned to match get_wmm_model_from_image signature
    return PyramidCoeffs, Histb, HistbBins, HistQuantiles, meanb, stdb

def generate_example(p):
    tmp = 256 * np.random.uniform(low=0, high=1, size=(3, p.resolution, p.resolution))

    out = sample_image(tmp, p, reconstruct_colors=p.correlate_channels)
    image = np.array((out - out.min()) / (out.max() - out.min()) * 255.0, dtype='uint8')

    return image

def generate_example_seed(seed, opt):
    random.seed(seed)
    np.random.seed(seed)

    if type(opt) == GenerationParams:
        p = opt
    else:
        p = GenerationParams(**opt.__dict__)
    if p.impose_color_model:
        p.colorPC, p.MeanColor, p.HistIma, p.HistImaBins, p.HistImaQuantiles, p.img = sample_colormodel(p)
    elif p.correlate_channels:
        p.colorPC, p.MeanColor, _, _, _, _ = sample_colormodel(p)

    if p.impose_spectrum:
        p.SpectrumMagnitude = sample_1f_spectrum(p)

    if p.impose_wmm:
        p.PyramidCoeffs, p.Histb, p.HistbBins, p.HistQuantiles, p.meanb, p.stdb = sample_wmm_model(p)

    image = generate_example(p)
    return image

def generate_single_example(file_number, output_folder, opt):
    img_output_folder = output_folder + '/' + str((file_number // 1000) * 1000).zfill(8)
    os.makedirs(img_output_folder, exist_ok=True)
    output_file = img_output_folder + '/' + str(file_number).zfill(8) + '.' + opt.output_file_type

    resolution = opt.resolution

    error_loading = False
    try:
        if os.path.exists(output_file):
            # open image, if it fails create again
            img = pil_loader(output_file)
            assert img.size == (resolution, resolution)
            return
    except:
        print("Error in an image {}, will regenerate!".format(file_number))
        error_loading = True
        pass

    if not os.path.exists(output_file) or error_loading:
        image = generate_example_seed(file_number, opt)

        # check again for race condition in case the script is running on multiple machines at the same time
        if not os.path.exists(output_file):
            cv2_imwrite(image, output_file)

def generate_dataset(opt, output_folder):
    file_numbers_to_generate = list(range(opt.samples))
    if opt.shuffle_generation:
        random.shuffle(file_numbers_to_generate)

    def generate_one(file_number):
        generate_single_example(file_number, output_folder, opt)

    if opt.parallel:
        p_map(generate_one, file_numbers_to_generate)
    else:
        for file_number in tqdm(file_numbers_to_generate):
            generate_one(file_number)


def large_scale_options(opt):
    opt.resolution = 256
    opt.samples = 1300000
    opt.Niter = 10
    return opt

def small_scale_options(opt):
    opt.resolution = 128
    opt.samples = 105000
    opt.Niter = 10
    return opt

def get_output_folder(opt):
    if len(opt.output_folder) == 0:
        output_folder = '../generated_datasets/{}/stat-'.format('small_scale' if opt.small_scale else 'large_scale')
        assert opt.small_scale or opt.large_scale, "When not using --small-scale or --large-scale the output path should be defined!"
        assert opt.Niter == 10, "Small/large scale parameters should be used with Niter == 10"
        # Use default names
        if opt.impose_spectrum and not opt.impose_wmm and not opt.impose_color_model and opt.correlate_channels:
            output_folder += 'spectrum/train'
        elif not opt.impose_spectrum and opt.impose_wmm and not opt.impose_color_model and opt.correlate_channels:
            output_folder += 'wmm/train'
        elif opt.impose_spectrum and not opt.impose_wmm and opt.impose_color_model and opt.correlate_channels:
            output_folder += 'spectrum_color/train'
        elif opt.impose_spectrum and opt.impose_wmm and opt.impose_color_model and opt.correlate_channels:
            output_folder += 'spectrum_color_wmm/train'
        else:
            raise Exception("Not one of the default modes, so output_path has to be passed as an argument!")
    else:
        output_folder = opt.output_folder + '/low_dimensional_parametric_image_model_{}'.format(opt.output_file_type)

        if opt.correlate_channels:
            output_folder += '_correlate_channels'
        output_folder += '_' + str(opt.resolution)

        if opt.impose_spectrum:
            output_folder += '_impose_spectrum'
        if opt.impose_color_model:
            output_folder += '_impose_color'
        if opt.impose_wmm:
            output_folder += '_impose_wmm'

    return output_folder


if __name__ == "__main__":
    # TODO: remove circularity by simply generating at higher ressolution and crop
    parser = argparse.ArgumentParser('Statistical image generation')

    parser.add_argument('--small-scale', action="store_true", help="use small-scale default parameters")
    parser.add_argument('--large-scale', action="store_true", help="use large-scale default parameters")

    parser.add_argument('--output-folder', default='', type=str, help='Encoder checkpoint to evaluate')
    parser.add_argument('--output-file-type', type=str, default='jpg', choices=['jpg', 'png'], help='Filetype to generate')

    parser.add_argument('--resolution', type=int, default=128, help='Resolution to use')
    parser.add_argument('--Niter', type=int, default=10, help='Number of iterations')
    parser.add_argument('--samples', type=int, default=105000, help='N samples to generate')
    parser.add_argument('--shuffle-generation', type=str2bool, default="False", help='Whether to generate samples in order or shuffled')

    parser.add_argument('--impose_spectrum', type=str2bool, default="False", help='Whether to impose the spectrum model or not')
    parser.add_argument('--impose_wmm', type=str2bool, default="False", help='Whether to impose the wmm model or not')
    parser.add_argument('--impose_color_model', type=str2bool, default="False", help='Whether to impose the color model or not')
    parser.add_argument('--correlate_channels', type=str2bool, default="True", help='Whether to apply a random rotation to the color channels so that they are correlated')

    parser.add_argument('--parallel', type=str2bool, default="False", help='Whether to apply a random rotation to the color channels so that they are correlated')

    opt = parser.parse_args()

    if opt.large_scale:
        opt = large_scale_options(opt)
    elif opt.small_scale:
        opt = small_scale_options(opt)

    output_folder = get_output_folder(opt)

    generate_dataset(opt, output_folder)
