import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorflow.contrib.rnn import *

class Model:
    '''
        One-shot learning
    '''
    def __init__(self, sess, name, batch_size, n_way, k_shot, training, use_fce, lr):
        self.sess = sess
        self.name = name
        self.batch_size = batch_size
        self.n_way = n_way  # class number
        self.k_shot = k_shot  # sample number
        self.training = training
        self.use_fce = use_fce
        self.lr = lr
        self.processing_steps = 10
        self._build_graph()

    def _build_graph(self):
        with tf.variable_scope('initialize_scope'):
            self.support_set_image = tf.placeholder(dtype=tf.float32, shape=[None, self.n_way * self.k_shot, 100, 50, 1], name='train_x')
            self.support_set_label = tf.placeholder(dtype=tf.int32, shape=[None, self.n_way * self.k_shot], name='train_y')
            self.example_image = tf.placeholder(dtype=tf.float32, shape=[None, 100, 50, 1], name='test_x')
            self.example_label = tf.placeholder(dtype=tf.int32, shape=[None], name='test_y')

        def _batch_norm(layer, act=tf.nn.elu, name='batch_norm'):
            with tf.variable_scope(name_or_scope=name, reuse=False if self.training else True):
                return BatchNormLayer(layer, act=act, is_train=self.training)

        def _image_encoder(image, reuse=False):
            with tf.variable_scope(name_or_scope='image_encoder', reuse=reuse):
                layer = InputLayer(inputs=image, name='input_layer')  # (100, 50)
                layer = Conv2d(layer=layer, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.identity, name='conv_01')
                layer = _batch_norm(layer=layer, act=tf.nn.elu, name='batch_norm_01')
                layer = PoolLayer(prev_layer=layer, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), pool=tf.nn.max_pool, name='max_pool_01')  # (50, 25)
                layer = Conv2d(layer=layer, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.identity, name='conv_02')
                layer = _batch_norm(layer=layer, act=tf.nn.elu, name='batch_norm_02')
                layer = PoolLayer(prev_layer=layer, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), pool=tf.nn.max_pool, name='max_pool_02')  # (25, 13)
                layer = Conv2d(layer=layer, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.identity, name='conv_03')
                layer = _batch_norm(layer=layer, act=tf.nn.elu, name='batch_norm_03')
                layer = PoolLayer(prev_layer=layer, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), pool=tf.nn.max_pool, name='max_pool_03')  # (13, 7)
                layer = Conv2d(layer=layer, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.identity, name='conv_04')
                layer = _batch_norm(layer=layer, act=tf.nn.elu, name='batch_norm_04')
                layer = PoolLayer(prev_layer=layer, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), pool=tf.nn.max_pool, name='max_pool_04')  # (7, 4)
                layer = Conv2d(layer=layer, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.identity, name='conv_05')
                layer = _batch_norm(layer=layer, act=tf.nn.elu, name='batch_norm_05')
                layer = PoolLayer(prev_layer=layer, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), pool=tf.nn.max_pool, name='max_pool_05')  # (4, 2)
            return ReshapeLayer(prev_layer=layer, shape=(-1, 1*1*64), name='encoder')

        def _fce_g(encoded_x_i):
            with tf.variable_scope('fce_g'):
                 layer = BiRNNLayer(prev_layer=encoded_x_i, cell_fn=BasicLSTMCell, n_hidden=32)
                 outputs, state_fw, state_bw = layer.outputs, layer.fw_final_state, layer.bw_final_state
            return tf.add(tf.stack(encoded_x_i.outputs), tf.stack(outputs))

        def _fce_f(encoded_x, g_embedding):
            cell = BasicLSTMCell(64)
            prev_state = cell.zero_state(self.batch_size, tf.float32)

            for step in range(self.processing_steps):
                output, state = cell(encoded_x.outputs, prev_state)

                h_k = tf.add(output, encoded_x.outputs)

                content_based_attention = tf.nn.softmax(tf.multiply(prev_state[1], g_embedding.outputs))
                r_k = tf.reduce_sum(tf.multiply(content_based_attention, g_embedding.outputs), axis=0)

                prev_state = LSTMStateTuple(state[0], tf.add(h_k, r_k))

            return output

        def _cosine_similarity(target, support_set):
            sup_similarity = []
            for i in UnStackLayer(support_set):
                i_normed = tf.nn.l2_normalize(i.outputs, 1)
                similarity = tf.matmul(tf.expand_dims(target.outputs, 1), tf.expand_dims(i_normed, 2))
                sup_similarity.append(similarity)
            return tf.squeeze(tf.stack(sup_similarity, axis=1))

        def _network():
            image_encoded = _image_encoder(self.example_image)
            support_set_image_encoded = StackLayer([_image_encoder(img) for img in tf.unstack(self.support_set_image, axis=1)])

            if self.use_fce:
                g_embedding = _fce_g(support_set_image_encoded)
                f_embedding = _fce_f(image_encoded, g_embedding)
            else:
                g_embedding = support_set_image_encoded
                f_embedding = image_encoded

            embeddings_similarity = _cosine_similarity(f_embedding, g_embedding)
            attention = tf.nn.softmax(embeddings_similarity)
            y_hat = tf.matmul(tf.expand_dims(attention, 1), tf.one_hot(self.support_set_label, self.n_way))
            logits = tf.squeeze(y_hat)

            return logits

        self.logits = _network()
        self.loss = tl.cost.cross_entropy(output=self.logits, target=self.example_label)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.pred = tf.argmax(self.logits, 1)

    def train(self, support_set_image, support_set_label, example_image, example_label):
        return self.sess.run([self.optimizer], feed_dict={self.support_set_image: support_set_image, self.support_set_label: support_set_label,
                                                          self.example_image: example_image, self.example_label: example_label})