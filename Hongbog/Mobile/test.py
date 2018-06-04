import tensorflow as tf
import numpy as np

x = tf.read_file('D:\\TensorflowMobile\\test\\right_6.png')
decode_img = tf.image.decode_png(x, channels=1, name='decode_img')
resize_img = tf.image.resize_images(decode_img, (80, 200))
transpose_img = tf.transpose(resize_img, perm=(1, 0, 2))

with tf.Session() as sess:
    d_img = sess.run(decode_img)
    r_img = sess.run(resize_img)
    t_img = sess.run(transpose_img)
    print(d_img.shape, r_img.shape, t_img.shape)
