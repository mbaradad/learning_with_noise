from vtab_eval.task_adaptation.adapt_and_eval import setup_environment_gpus
setup_environment_gpus(1)
import tensorflow as tf

from vtab_eval.port_model.port_utils import *

results_path = '/data/vision/torralba/movies_sfm/home/no_training_cnn/contrastive_image_models/vtab_eval/results'

def list_saved_model(path):
    from tensorflow.core.protobuf.saved_model_pb2 import SavedModel

    saved_model = SavedModel()
    with open(path, 'rb') as f:
        saved_model.ParseFromString(f.read())
    model_op_names = set()


    # Iterate over every metagraph in case there is more than one
    for i, meta_graph in enumerate(saved_model.meta_graphs):
        # Add operations in the graph definition
        model_op_names.update(node.name for node in meta_graph.graph_def.node)
        # Go through the functions in the graph definition
        for func in meta_graph.graph_def.library.function:
            # Add operations in each function
            model_op_names.update(node.name for node in func.node_def)
    # Convert to list, sorted if you want
    model_o_names = sorted(model_op_names)
    print(*model_o_names, sep='\n')

def list_parameters():
    from tensorflow.python.training import checkpoint_utils as cp

    model_path = '/data/vision/torralba/movies_sfm/home/no_training_cnn/contrastive_image_models/vtab_eval/checkpoints_to_test/sup-100-rotation'
    list_saved_model(model_path + '/saved_model.pb')
    checkpoint = tf.train.latest_checkpoint()

    print(cp.list_variables(checkpoint))

    return

def test_same_parameters():
    model_name = 'moco_v2_pretrained_200_linear_only_1k_samples'
    model_name = 'moco_v2_imagenet_pytorch_all_samples'
    model_path = results_path + '/{}'.format(model_name) + '/{}'


    model_path_0 = model_path.format('caltech101')
    model_path_1 = model_path.format('cifar(num_classes=100)')

    model_path_0 = 'temp/moco_v2_pretrained_full'
    model_path_1 = 'temp/moco_v2_pretrained_linear_only'

    latest_0 = tf.train.latest_checkpoint(model_path_0)
    latest_1 = tf.train.latest_checkpoint(model_path_1)

    original_0 = '-'.join([*latest_0.split('-')[:-1], str(0)])
    original_1 = '-'.join([*latest_1.split('-')[:-1], str(0)])


    for v_name in get_variables(model_path_0):
        if 'global_step' == v_name:
            continue
        import numpy as np
        v_0 = load_variable(original_1, v_name)
        v_1 = load_variable(latest_1, v_name)
        if not 'linear' in v_name:
            assert np.allclose(v_0, v_1), "Variable {} has been modified, while".format(v_name)
        else:
          assert not np.allclose(v_0, v_1)

if __name__ == '__main__':
    test_same_parameters()