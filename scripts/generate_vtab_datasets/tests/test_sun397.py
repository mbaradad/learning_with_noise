import resource

# to avoid ancdata error for too many open files, same as ulimit in console
# maybe not necessary, but doesn't hurt
resource.setrlimit(resource.RLIMIT_NOFILE, (131072, 131072))

from modified_builders.sun397_ours import Sun397_ours

builder_kwargs = {'config': 'tfds',
                  'version': '4.*.*',
                  'data_dir': '/data/vision/torralba/scratch/mbaradad/tensorflow_datasets_test'}
dataset_builder = Sun397_ours(**builder_kwargs)

dataset_builder.download_and_prepare()

