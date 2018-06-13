import tensorflow as tf

# MNIST 데이터 임포트
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('D:\\Source\\PythonRepository\\LearningTensorFlow\\Chapter5_Text_Sequence_Tensorboard\\mnist_data', one_hot=True)

# 매개변수 정의
element_size = 28
time_step = 28
num_classes = 10
batch_size = 128
hidden_layer_size = 128

# 텐서보드 모델 요약을 저장할 위치
LOG_DIR = 'D:\\Source\\PythonRepository\\LearningTensorFlow\\Chapter5_Text_Sequence_Tensorboard\\logs\\RNN_with_summaries'

# 입력과 레이블을 위한 placeholder 생성
_inputs = tf.placeholder(dtype=tf.float32, shape=[None, time_step, element_size], name='inputs')
y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name='labels')

# 요약을 로깅하는 몇몇 연산을 추가하는 헬퍼 함수
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

with tf.name_scope('rnn_weights'):
    with tf.name_scope('W_x'):
        Wx = tf.Variable(tf.zeros([element_size, hidden_layer_size]))
        variable_summaries(Wx)
    with tf.name_scope('W_h'):
        Wh = tf.Variable(tf.zeros([hidden_layer_size, hidden_layer_size]))
        variable_summaries(Wh)
    with tf.name_scope('Bias'):
        b_rnn = tf.Variable(tf.zeros([hidden_layer_size]))
        variable_summaries(b_rnn)

def rnn_step(previous_hidden_state, x):
    current_hidden_size = tf.tanh(tf.matmul(previous_hidden_state, Wh) + tf.matmul(x, Wx) + b_rnn)
    return current_hidden_size

processed_input = tf.transpose(_inputs, perm=[1, 0, 2])

initial_hidden = tf.zeros([batch_size, hidden_layer_size])

all_hidden_states = tf.scan(rnn_step,
                            processed_input,
                            initializer=initial_hidden,
                            name='states')

with tf.name_scope('linear_layer_weights') as scope:
    with tf.name_scope('W_linear'):
        Wl = tf.Variable(tf.truncated_normal([hidden_layer_size,
                                              num_classes],
                                             mean=0,
                                             stddev=.01))
        variable_summaries(Wl)

    with tf.name_scope('Bias_linear'):
        bl = tf.Variable(tf.truncated_normal([num_classes], mean=0, stddev=.01))
        variable_summaries(bl)

def get_linear_layer(hidden_state):
    return tf.matmul(hidden_state, Wl) + bl

with tf.name_scope('linear_layer_weights') as scope:
    all_outputs = tf.map_fn(get_linear_layer, all_hidden_states)
    output = all_outputs[-1]
    tf.summary.histogram('outputs', output)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    opt = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)

with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float64)) * 100
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

test_data = mnist.test.images[:batch_size].reshape((-1, time_step, element_size))
test_label = mnist.test.labels[:batch_size]

with tf.Session() as sess:
    # LOG_DIR 에 텐서보드에서 사용할 요약을 기록
    train_writer = tf.summary.FileWriter(LOG_DIR + '\\train', graph=tf.get_default_graph())
    test_writer = tf.summary.FileWriter(LOG_DIR + '\\test', graph=tf.get_default_graph())

    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
        batch_x = batch_x.reshape((batch_size, time_step, element_size))
        summary, _ = sess.run([merged, opt], feed_dict={_inputs: batch_x, y: batch_y})

        train_writer.add_summary(summary=summary, global_step=i)

        if i % 1000 == 0:
            tr_acc, tr_loss = sess.run([accuracy, loss], feed_dict={_inputs: batch_x, y: batch_y})
            print('Iter ' + str(i) + ', Minibatch Loss={:.6f}, Training Accuracy={:.5f}'.format(tr_loss, tr_acc))

        if i % 10:
            summary, ts_acc = sess.run([merged, accuracy], feed_dict={_inputs: test_data, y: test_label})
            test_writer.add_summary(summary=summary, global_step=i)

        ts_acc = sess.run(accuracy, feed_dict={_inputs: test_data, y: test_label})
    print('Test Accuracy:', ts_acc)