import tensorflow as tf

rnn = tf.contrib.rnn
slim = tf.contrib.slim

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
            self.support_set_image = tf.placeholder(dtype=tf.float32, shape=[None, self.n_way * self.k_shot, 28, 28, 1], name='train_x')
            self.support_set_label = tf.placeholder(dtype=tf.int32, shape=[None, self.n_way * self.k_shot], name='train_y')
            self.example_image = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='test_x')
            self.example_label = tf.placeholder(dtype=tf.int32, shape=[None], name='test_y')

        def _image_encoder(image):
            with slim.arg_scope([slim.conv2d], num_outputs=64, kernel_size=3, normalizer_fn=slim.batch_norm):
                net = slim.conv2d(image)
                net = slim.max_pool2d(net, [2, 2])
                net = slim.conv2d(net)
                net = slim.max_pool2d(net, [2, 2])
                net = slim.conv2d(net)
                net = slim.max_pool2d(net, [2, 2])
                net = slim.conv2d(net)
                net = slim.max_pool2d(net, [2, 2])
            return tf.reshape(net, [-1, 1 * 1 * 64])

        def _fce_g(encoded_x_i):
            with tf.variable_scope('fce_g'):
                fw_cell = rnn.BasicLSTMCell(32)  # 32 is half of 64 (output from cnn)
                bw_cell = rnn.BasicLSTMCell(32)
                outputs, state_fw, state_bw = rnn.static_bidirectional_rnn(fw_cell, bw_cell, encoded_x_i, dtype=tf.float32)

            return tf.add(tf.stack(encoded_x_i), tf.stack(outputs))

        def _fce_f(encoded_x, g_embedding):
            cell = rnn.BasicLSTMCell(64)
            prev_state = cell.zero_state(self.batch_size, tf.float32)

            for step in range(self.processing_steps):
                output, state = cell(encoded_x, prev_state)

                h_k = tf.add(output, encoded_x)

                content_based_attention = tf.nn.softmax(tf.multiply(prev_state[1], g_embedding))
                r_k = tf.reduce_sum(tf.multiply(content_based_attention, g_embedding), axis=0)

                prev_state = rnn.LSTMStateTuple(state[0], tf.add(h_k, r_k))

            return output

        def _cosine_similarity(target, support_set):
            sup_similarity = []
            for i in tf.unstack(support_set):
                i_normed = tf.nn.l2_normalize(i, 1)
                similarity = tf.matmul(tf.expand_dims(target, 1), tf.expand_dims(i_normed, 2))
                sup_similarity.append(similarity)
            return tf.squeeze(tf.stack(sup_similarity, axis=1))

        def _network():
            image_encoded = _image_encoder(self.example_image)
            support_set_image_encoded = [_image_encoder(img) for img in tf.unstack(self.support_set_image, axis=1)]

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
        self.loss = tf.losses.sparse_softmax_cross_entropy(self.example_label, self.logits)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.pred = tf.argmax(self.logits, 1)

    def train(self, support_set_image, support_set_label, example_image, example_label):
        return self.sess.run([self.logits, self.pred, self.loss, self.optimizer],
                             feed_dict={self.support_set_image: support_set_image, self.support_set_label: support_set_label,
                                        self.example_image: example_image, self.example_label: example_label})

    def test(self, support_set_image, support_set_label, example_image):
        return self.sess.run([self.logits, self.pred], feed_dict={self.support_set_image: support_set_image, self.support_set_label: support_set_label,
                                                                  self.example_image: example_image})