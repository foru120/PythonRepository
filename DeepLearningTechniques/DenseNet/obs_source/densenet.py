import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

layers = tf.contrib.layers
arg_scope = tf.contrib.framework.arg_scope

from DeepLearningTechniques.DenseNet.obs_source.ops import *
from DeepLearningTechniques.DenseNet.obs_source.utils import *

class Dense10(object):
	def __init__(self, sess, num_classes=10, depth=40):
		self.sess = sess
		self.model_name = 'GAN'
		self.epoch = FLAGS.epoch
		self.batch_size = FLAGS.batch_size
		
		self.depth = depth
		self.layers = int((self.depth - 4) / 3)
		self.growth = FLAGS.growth_rate
		self.d_rate = FLAGS.drop_rate

		self.lr = FLAGS.init_rate
		self.decay_rate = 0.96
		self.decay_step = 50000*FLAGS.decay_step // self.batch_size
		self.iteration = (self.epoch * 50000) // self.batch_size
		
		self.num_classes = num_classes
		
		self.im_size = 32
		self.im_depth = 3
		
		self.num_gpus = FLAGS.num_gpus
		self.regularizer = layers.l2_regularizer(scale=0.1)
		self._Build()

	def _DenseNet(self, inputs, num_classes=10, is_training=True, reuse=False):
		with tf.variable_scope('Dense_', reuse=reuse):
			with arg_scope([layers.conv2d], activation_fn=None, normalizer_fn=None, kernel_size=[3, 3], stride=[1, 1]):
				with arg_scope([layers.batch_norm], is_training=is_training, renorm=False):
					with arg_scope([layers.dropout], keep_prob=self.d_rate, is_training=is_training):
						net = layers.conv2d(inputs, 16, scope='init_conv1')
						net, filters = BlockCLayers(net, self.layers, 16, self.growth)
						net = layers.avg_pool2d(net, [2, 2]) ## 32*32 to 16*16
						net, filters = BlockCLayers(net, self.layers, filters, self.growth)
						net = layers.avg_pool2d(net, [2, 2]) ## 16*16 to 8*8
						net, filters = BlockCLayers(net, self.layers, filters, self.growth)
						
						net = lrelu(layers.batch_norm(net))
						net = layers.avg_pool2d(net, [8, 8])
						
						logits = layers.conv2d(net, num_classes, weights_regularizer=self.regularizer)
						logits = tf.squeeze(logits)
					
						return logits
			
	def _Build(self):
		with tf.device('/gpu:%s' % self.num_gpus):
			#self.inputs_node = tf.placeholder(tf.float32, [None, 32*32*3] )
			#self.labels_node = tf.placeholder(tf.float32, [None, self.num_classes] )
			#in_images = tf.reshape(self.inputs_node, [-1, 32, 32, 3])
			
			images, labels = cifar10.distorted_inputs()
			images_eval, labels_eval = cifar10.inputs(eval_data=True)
			
			logits = self._DenseNet(images, self.num_classes, is_training=True, reuse=False)
			logits_eval = self._DenseNet(images_eval, self.num_classes, is_training=False, reuse=True)
			#probs = tf.nn.softmax(logits)
			
			labels = dense_to_one_hot(labels, self.num_classes)
			
			step = tf.Variable(0, name='global_step', trainable=False)
			lr = tf.train.exponential_decay(self.lr,
			                                global_step=step,
			                                decay_steps=self.decay_step,
			                                decay_rate=self.decay_rate,
			                                staircase=True)
			
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
			reg_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
			reg_term = layers.apply_regularization(self.regularizer, reg_vars)
			
			self.loss += reg_term
			
			vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Dense_')
			
			with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
				self.opt = tf.train.AdamOptimizer(lr).minimize(self.loss, var_list=vars)
			
			correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
			
			self.sess.run(tf.global_variables_initializer())
	
	def _Training(self):
		
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		
		for iter in range(self.iteration):
			_, _loss, train_acc = self.sess.run([self.opt, self.loss, self.accuracy])
			
			if iter % 100 == 0:
				print('>>> Iteration:[%d/%d] loss:%.8f avg_accuracy:%.4f' % (iter, self.iteration, _loss, train_acc))

			"""
			if iter % 500 == 0 or iter + 1 == self.iteration:
				samples = self.sess.run(self.fake_sample, feed_dict={self.z: self.sample_z})
				save_dir = './' + self.model_name + '/Images/' + str(iter).zfill(10) + '.jpeg'
				save_image(samples, save_dir)
			"""
		coord.request_stop()
		coord.join(threads)