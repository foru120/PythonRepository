import os

import tensorflow as tf
from DeepLearningTechniques.DenseNet.densenet import Dense10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train', """""")
tf.app.flags.DEFINE_string('num_gpus', '0', """GPU ID """)

tf.app.flags.DEFINE_integer('epoch', 100, """""")
tf.app.flags.DEFINE_integer('batch_size', 128, """""")

tf.app.flags.DEFINE_integer('decay_step', 2, """""")

tf.app.flags.DEFINE_integer('growth_rate', 12, """""")
tf.app.flags.DEFINE_float('init_rate', 1e-4, """""")
tf.app.flags.DEFINE_float('drop_rate', 0.8, """""")

def main(_):
	cifar10.maybe_download_and_extract()
	if tf.gfile.Exists(FLAGS.train_dir):
		tf.gfile.DeleteRecursively(FLAGS.train_dir)
	tf.gfile.MakeDirs(FLAGS.train_dir)
	
	os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.num_gpus
	config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

	with tf.Session(config=config) as sess:
		Net = Dense10(sess, num_classes=10, depth=40)
		Net._Training()
if __name__ == '__main__':
	#main()
	tf.app.run()
