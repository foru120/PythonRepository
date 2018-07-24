import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time
from Hongbog.DementiaDiagnosis.wt_mobilenet import mobilenet_optimizer, mobilenet_losses, mobilenet_v1, training_scope
import Hongbog.DementiaDiagnosis.load_pickle as HongBog

''' 수정시 변경해야할 변수 LIST'''
# 4, 37, 48, 49, 62, 78, 122 lines.

''' Global Variable '''
NxNSize = 192
train_batch = 10
test_batch = 10
epochs = 100
model = '192x192_gelontoxon_rgb'

tot_data, tot_label = HongBog.load_training_data()
train_data, train_label = tot_data[:40], tot_label[:40]
test_data, test_label = tot_data[40:], tot_data[40:]

with slim.arg_scope(training_scope(is_training=True)):
    images = tf.placeholder(tf.float32, [None, NxNSize, NxNSize, 3])
    classes = tf.placeholder(tf.float32, [None, 2])
    logits, end_points = mobilenet_v1(images)

total_loss = mobilenet_losses(logits, classes)
train_op = mobilenet_optimizer(total_loss, step=epochs)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(classes, 1)), dtype=tf.float32))

# ema = tf.train.ExponentialMovingAverage(0.999)
# vars = ema.variables_to_restore()
# saver = tf.train.Saver(vars)
saver = tf.train.Saver()

g = tf.get_default_graph()
with tf.Session(graph=g) as sess:

    sess.run(tf.global_variables_initializer())
    print('>> MobileNet initialized')

    for step in range(epochs):
        stime = time.time()
        train_acc = []
        loss = 0.

        for start in range(0, len(train_data), train_batch):
            batch_x, batch_y = train_data[start:start+train_batch], train_label[start:start+train_batch]
            a, l, _ = sess.run([accuracy, total_loss, train_op], feed_dict={images: batch_x, classes: batch_y})
            train_acc.append(a)
            loss += l / train_batch
        etime = time.time()
        acc = np.mean(np.array(train_acc))
        print('>> epoch : {}, loss : {}, accuracy : {}, time : {}'.format(step+1, loss, acc, round(etime-stime, 3)))
    print('>> Learning Finished')

    tf.train.write_graph(sess.graph_def, 'tmp/', 'graph.pbtxt')
    saver.save(sess, 'log/' + model + '.ckpt')
