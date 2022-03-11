import numpy as np

import tensorflow as tf
tf.executing_eagerly()

import tensorflow_hub as hub

from vtab_eval.port_model.port_model import load_tf_model

from scipy.special import softmax

from vtab_eval.task_adaptation.adapt_and_eval import setup_environment_gpus
setup_environment_gpus(2)
from vtab_eval.port_model.port_model import load_image_tf, load_image

from my_python_utils.common_utils import *

models_to_port = {'resnet_imagenet_supervised',
                  'resnet_imagenet_contrastive',
                  'dead_leaves_contrastive'}



def printTensors(pb_file):

    # read pb into graph_def
    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import graph_def
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    # print operations
    for op in graph.get_operations():
        print(op.name)

def test_ported_imagenet_same_values():
    model = load_tf_model('vtab_eval/checkpoints_to_test/resnet50_pretrained', 'serving_default')

    from torchvision.models import resnet50
    resnet_pytorch = resnet50(pretrained=True).cuda()

    import torchvision.transforms as transforms

    normalize_transform = transforms.Normalize(mean=(0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
                                               std=(0.26826768628079806, 0.2610450402318512, 0.26866836876860795))

    import tensorflow as tf

    img = load_image('vtab_eval/port_model/dog.jpeg')
    img_torch = totorch(img).cuda()
    tf_img = tf.convert_to_tensor(img.transpose((1, 2, 0))[None])
    pytorch_values = resnet_pytorch(normalize_transform(img_torch)[None,...])
    tf_values = model(tf_img)

def test_imagenet_supervised_model_class(handle, signature):
    model = load_tf_model(handle, signature)

    #keras_hub_model = hub.KerasLayer(handle)
    #model0 = tf.keras.final_models.Sequential(keras_hub_model)
    #model0.build(input_shape=[1,224,224,3])

    # variable_names = sorted([k.name for k in keras_hub_model.variables])
    # The input images are expected to have color values in the range [0,1].
    # ^^ which means that the module includes the normalization transform.

    x = load_image_tf('vtab_eval/port_model/dog.jpeg')

    model_out = model(x)
    try:
        logits = np.array(model_out['logits'])[0, 1:] # first logit is background for hub final_models, see the hub
    except:
        logits = np.array(model_out['output_0'])[0]

    #sess = tf.Session()
    #op = sess.graph.get_operations()
    #[m.values() for m in op][1]

    assert len(logits) == 1000, "N logits != 1000, check the model is imagenet classification or and that it includes the background class"
    probs = softmax(logits)
    detected_class_id = logits.argsort()[-5:][::-1]

    import urllib
    id_to_class_name = pickle.load(urllib.request.urlopen(
        'https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl'))
    for k in detected_class_id:
        print("{}: {}".format(float2str(probs[k], 2), id_to_class_name[k]) )
    correct_class = 258
    assert detected_class_id[0] == correct_class, "Failed to detect a {}".format(id_to_class_name[correct_class])


if __name__ == '__main__':
    # list_saved_model('vtab_eval/checkpoints_to_test/resnet50_pretrained/saved_model.pb')
    handle, signature = [('https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/3', 'image_feature_vector'),
                         ('https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5', 'serving_default'),
                         ('vtab_eval/port_model/hub_models/imagenet_resnet_v2_50_classification_5', 'serving_default'),
                         ('vtab_eval/checkpoints_to_test/resnet50_pretrained', 'serving_default'),
                         ('vtab_eval/checkpoints_to_test/resnet50_pretrained_with_class_this_is_a_test', 'serving_default')][3]
    #test_ported_imagenet_same_values()
    test_imagenet_supervised_model_class(handle, signature)
