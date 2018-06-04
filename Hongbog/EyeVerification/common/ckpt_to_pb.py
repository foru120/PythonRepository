import tensorflow.contrib.slim as slim
import tensorflow as tf

with tf.Session(graph=tf.Graph) as sess:
    saver = tf.train.Saver()
    saver.restore(sess, 'D:\\TensorflowMobile\\pb_files\\inception_resnet_v2_2016_08_30\\inception_resnet_v2_2016_08_30.ckpt')

    tf.train.write_graph(sess.graph_def, 'D:\\TensorflowMobile\\pb_files\\inception_resnet_v2_2016_08_30', 'graph.pbtxt')
    print('>> Graph saved')

    builder = tf.saved_model.builder.SavedModelBuilder('D:\\TensorflowMobile\\pb_files\\inception_resnet_v2_2016_08_30\\sample')
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
    builder.save()