import tensorflow as tf

layers = tf.contrib.layers
arg_scope = tf.contrib.framework.arg_scope

def lrelu(x, leak=0.2, name='lrelu'):
	return tf.maximum(x, leak*x)

def BlockCLayers(inputs, L, in_filters, Growth):
	c = inputs
	filters = in_filters
	for idx in range(L):
		net = lrelu(layers.batch_norm(inputs))
		net = layers.conv2d(net, in_filters)
		net = layers.dropout(net)
		
		c = tf.concat([net, c], axis=3)
		filters += Growth
	return c, filters