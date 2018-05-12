def train_lr_flip_data(self, x, y):
    with tf.variable_scope('train_lr_flip_data'):
        x = tf.read_file(x)
        x = tf.image.decode_png(x, channels=1, name='decode_img')
        x = tf.image.resize_images(x, size=(160, 50))
        x = tf.image.random_flip_left_right(x)
        x = tf.subtract(tf.divide(x, 127.5), 1)
    return x, y


def train_ud_flip_data(self, x, y):
    with tf.variable_scope('train_ud_flip_data'):
        x = tf.read_file(x)
        x = tf.image.decode_png(x, channels=1, name='decode_img')
        x = tf.image.resize_images(x, size=(160, 50))
        x = tf.image.random_flip_up_down(x)
        x = tf.subtract(tf.divide(x, 127.5), 1)
    return x, y


def train_rot_data(self, x, y):
    with tf.variable_scope('train_rot_data'):
        x = tf.read_file(x)
        x = tf.image.decode_png(x, channels=1, name='decode_img')
        x = tf.image.resize_images(x, size=(160, 50))
        x = tf.image.rot90(x, k=tf.random_uniform([], maxval=4, dtype=tf.int32))
        x = tf.subtract(tf.divide(x, 127.5), 1)
    return x, y