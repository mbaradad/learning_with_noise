from tensorflow.python.eager import context
import tensorflow as tf

def load_variable(model_dir, name):
    with context.graph_mode():
        return tf.train.load_variable(model_dir, name)


def get_variables(model_dir):
    with context.graph_mode():
        return [name for name, _ in tf.train.list_variables(model_dir)]