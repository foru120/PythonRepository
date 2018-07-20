import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from Hongbog.DementiaDiagnosis.wt_mobilenet import mobilenet_v1, training_scope
import Hongbog.DementiaDiagnosis.load_pickle as HongBog

NxNSize = 192
model = '192x192_gelontoxon_rgb'
batch_size = 10
accuracy = []

with slim.arg_scope(training_scope(is_training=False)):
    images = tf.placeholder(tf.float32, [None, NxNSize, NxNSize, 3])
    classes = tf.placeholder(tf.float32, [None, 2])
    logits, end_points = mobilenet_v1(images)

# ema = tf.train.ExponentialMovingAverage(0.999)
# vars = ema.variables_to_restore()
# saver = tf.train.Saver(vars)
saver = tf.train.Saver()

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(classes, 1)), dtype=tf.float32))

g = tf.get_default_graph()
with tf.Session(graph=g) as sess:
    # sess.run(tf.global_variables_initializer())

    saver.restore(sess, 'log/' + model + '.ckpt')

    test_data, test_label = HongBog.load_training_data()
    acc = []
    for start_idx in range(0, len(test_data), batch_size):
        sample_size = len(test_data) if batch_size > len(test_data) else batch_size
        test_x_batch, test_y_batch = test_data[start_idx:start_idx + sample_size],\
                                     test_label[start_idx:start_idx + sample_size]

        a = sess.run([accuracy], feed_dict={images: test_x_batch, classes: test_y_batch})
        acc.append(a)

    print('Test Accuracy : {}%'.format(np.mean(np.asarray(acc))))

    tf.train.write_graph(sess.graph, 'tmp/', 'mobilenetV1_graph_test.pbtxt')
    saver.save(sess, 'log/test_' + model + '.ckpt')

