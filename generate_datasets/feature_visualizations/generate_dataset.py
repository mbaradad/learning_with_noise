import sys, os
sys.path.append('..')

assert os.getcwd().endswith('generate_datasets'), "This script should be run from generate_datasets folder."

import torchvision.models as models

from torchvision.datasets.folder import pil_loader

from utils_generation import *

import time, datetime
import torch

EMBEDDING_SIZE=2048
def generate_sorted_k_of_n(N):
    to_generate = []

    for i in range(EMBEDDING_SIZE):
        if len(to_generate) >= N: return to_generate
        to_generate.append([i])
    for i in range(EMBEDDING_SIZE):
        for j in range(i + 1, EMBEDDING_SIZE):
            if len(to_generate) >= N: return to_generate
            to_generate.append([i, j])
    for i in range(EMBEDDING_SIZE):
        for j in range(i + 1, EMBEDDING_SIZE):
            for k in range(j + 1, EMBEDDING_SIZE):
                if len(to_generate) >= N: return to_generate
                to_generate.append([i, j])

def generate_dataset(generation_stuff, N):
    to_generate = generate_sorted_k_of_n(N)

    assert len(to_generate) == N
    random.seed(time.time())
    random.shuffle(to_generate)
    generate_n(generation_stuff, to_generate)

def generate_n(generation_stuff, positions_to_generate, save=True, random_code=False):
    encoder = generation_stuff['model']

    import feature_visualizations.vis.grad_ascent.lucent.render as render
    from lucent.modelzoo.util import get_model_layers
    model_layers = get_model_layers(encoder)

    block_name = 'avgpool'

    i = 0
    pbar = tqdm(total=len(positions_to_generate))
    threaded_writer = ThreadeImageWriter(queue_size=10, use_threading=generation_stuff['threaded_writer'])
    while i < len(positions_to_generate):
        actual_filters_names = []
        actual_filters_files = []
        while i < len(positions_to_generate) and len(actual_filters_names) < generation_stuff['batch_size']:
            filter_js = positions_to_generate[i]
            filters_string = '_'.join([str(k).zfill(4) for k in filter_js])

            if not random_code:
                filter_name = "{}:{}".format(block_name, filters_string)
                feature_visualization_folder = generation_stuff['output_folder'] + '/layer_{}_{}'.format(block_name,
                                                                                                          str(filter_js[0]).zfill(4))
            else:
                filter_name = "{}:random".format(block_name)
                feature_visualization_folder = 'temp' # TODO if we end up generating this dataset
            if save:
                os.makedirs(feature_visualization_folder,exist_ok=True)
            feature_visualization_file = feature_visualization_folder + '/{}.jpg'.format(filters_string)

            i += 1
            pbar.update(1)
            error = False
            try:
                if os.path.exists(feature_visualization_file):
                    # open image, if it fails create again
                    img = pil_loader(feature_visualization_file)
                    assert img.size == (256, 256)
                    continue
            except:
                print("Error in an image, will regenerate!")
                error = True
                pass

            if error or not os.path.exists(feature_visualization_file):
                actual_filters_names.append(filter_name)
                actual_filters_files.append(feature_visualization_file)
        try:
            feat_imgs, losses_per_iteration = render.render_vis(encoder, actual_filters_names, thresholds=(generation_stuff['n_iters'],), show_inline=False,
                                                  show_image=False, pool_mode='mean',
                                                  negative_activation=False, fixed_image_size=256,
                                                  break_on_error=True, debug=generation_stuff['debug'], fft=True, progress=True)
            feat_imgs = feat_imgs[-1].transpose((0, 3, 1, 2))
        except Exception as e:
            print(e)
            continue
        if save:
            threaded_writer.put_images_and_files(feat_imgs, actual_filters_files)


def setup_generation_stuff(args):
    model = models.__dict__['resnet50']()
    loaded = False
    checkpoint_file = args.checkpoint

    if not args.random_init and os.path.isfile(checkpoint_file):
        print("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file, map_location="cpu")

        # rename moco_training pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        args.start_epoch = 0
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        loaded = True

        print("=> loaded pre-trained model '{}'".format(checkpoint_file))


    if not loaded and not args.random_init:
        raise Exception("Checkpoint not loaded!")

    device = "cuda:0"
    model = model.to(device).eval()

    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    if len(args.output_folder) == 0:
        if args.small_scale or args.large_scale:
            output_folder = '../generated_datasets/{}/feature_vis-'.format('small_scale' if opt.small_scale else 'large_scale')
            if args.random_init:
                output_folder += 'random/train'
            elif checkpoint_file == '../encoders/large_scale/dead_leaves-mixed/checkpoint_0199.pth.tar':
                output_folder += 'dead_leaves/train'

        else:
            raise Exception("Output folder should be a parameter when not using --large-scale or --small-scale")
    else:
        output_folder = args.output_folder


    return {"model": model,
            "output_folder": output_folder,
            "debug": args.debug,
            "n_iters": args.n_iters,
            "batch_size":  args.batch_size,
            "threaded_writer": args.threaded_writer}

def large_scale_options(opt):
    opt.resolution = 256
    opt.samples = 1300000
    return opt

def small_scale_options(opt):
    # both small and large scale are at 256
    opt.resolution = 256
    opt.samples = 105000
    return opt

def parse_option():
    parser = argparse.ArgumentParser('Visualize activations')

    parser.add_argument('--small-scale', action="store_true", help="use small-scale default parameters")
    parser.add_argument('--large-scale', action="store_true", help="use large-scale default parameters")

    parser.add_argument('--checkpoint', type=str, help='Encoder checkpoint to evaluate', default='../encoders/large_scale/dead_leaves-mixed/checkpoint_0199.pth.tar')
    parser.add_argument('--random-init', action='store_true', help="Use random initialization instead of precomputed checkpoint!")

    parser.add_argument('--output-folder', default='', type=str, help="Use random initialization instead of precomputed checkpoint!")

    parser.add_argument('--samples', type=int, default=105000)

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n-iters', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=64)

    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--threaded-writer', type=str2bool, default=True)

    opt = parser.parse_args()
    select_gpus(str(opt.gpu))

    opt.gpu = torch.device('cuda', 0)

    return opt


if __name__ == '__main__':
    opt = parse_option()

    if opt.large_scale:
        opt = large_scale_options(opt)
    elif opt.small_scale:
        opt = small_scale_options(opt)

    generation_stuff = setup_generation_stuff(opt)

    generate_dataset(generation_stuff, opt.samples)

