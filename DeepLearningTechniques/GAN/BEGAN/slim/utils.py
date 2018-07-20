import tensorflow as tf
from data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_name',
    'celeba',
    'The name of the dataset prefix.'
)

tf.app.flags.DEFINE_string(
    'dataset_dir',
    '/home/kyh/dataset/img_celeba',
    'A directory containing a set of subdirectories representing class names. '
    'Each subdirectory should contain PNG or JPG encoded images.'
)

tf.app.flags.DEFINE_string(
    'num_shards',
    5,
    'A number of sharding for TFRecord files(integer).'
)

tf.app.flags.DEFINE_string(
    'ratio_val',
    0.2,
    'A ratio of validation datasets for TFRecord files(float, 0.~1.).'
)