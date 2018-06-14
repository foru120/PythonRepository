import numpy as np
import tensorflow as tf

#todo 상수 정의
batch_size = 128
embedding_dimension = 64
num_classes = 2
hidden_layer_size = 32
times_steps = 6
element_size = 1

#todo 제로 패딩 수행
digit_to_word_map = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'FIve', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine'}
digit_to_word_map[0] = 'PAD'

even_sentences = []
odd_sentences = []
seqlens = []

for i in range(10000):
    rand_seq_len = np.random.choice(range(3, 7))
    seqlens.append(rand_seq_len)
    rand_odd_ints = np.random.choice(range(1, 10, 2), rand_seq_len)
    rand_even_ints = np.random.choice(range(2, 10, 2), rand_seq_len)

    # 패딩
    if rand_seq_len < 6:
        rand_odd_ints = np.append(rand_odd_ints, [0] * (6 - rand_seq_len))
        rand_even_ints = np.append(rand_even_ints, [0] * (6 - rand_seq_len))

    odd_sentences.append(' '.join([digit_to_word_map[r] for r in rand_odd_ints]))
    even_sentences.append(' '.join([digit_to_word_map[r] for r in rand_even_ints]))

data = even_sentences + odd_sentences
seqlens *= 2  # 홀수, 짝수 문장 길이

# 단어를 인덱스에 매핑
word2index_map = {}
index = 0

for sent in data:
    for word in sent.lower().split():
        if word not in word2index_map:
            word2index_map[word] = index
            index += 1
# 역방향 매핑
index2word_map = {index: word for word, index in word2index_map.items()}
vocabulary_size = len(index2word_map)

#todo 레이블 생성 후 학습/테스트 데이터로 분할
labels = [1] * 10000 + [0] * 10000
for i in range(len(labels)):
    label = labels[i]
    one_hot_encoding = [0] * 2
    one_hot_encoding[label] = 1
    labels[i] = one_hot_encoding

data_indices = list(range(len(data)))
np.random.shuffle(data_indices)
data = np.asarray(data)[data_indices]
labels = np.asarray(labels)[data_indices]
seqlens = np.asarray(seqlens)[data_indices]

train_x = data[:10000]
train_y = labels[:10000]
train_seqlens = seqlens[:10000]

test_x = data[10000:]
test_y = labels[10000:]
test_seqlens = seqlens[10000:]

#todo 문장의 일괄 처리 데이터를 생성하는 함수 (mini-batch)
def get_sentence_batch(batch_size, data_x, data_y, data_seqlens):
    instance_indices = list(range(len(data_x)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [[word2index_map[word] for word in data_x[i].lower().split()] for i in batch]
    y = [data_y[i] for i in batch]
    seqlens = [data_seqlens[i] for i in batch]
    return x, y, seqlens

#todo 데이터에 사용할 placeholder 생성
_inputs = tf.placeholder(tf.int32, shape=[batch_size, times_steps])
_labels = tf.placeholder(tf.float32, shape=[batch_size, num_classes])

# 동적 계산을 위한 seqlens
_seqlens = tf.placeholder(tf.int32, shape=[batch_size])

#todo 단어ID 를 tf.nn.embedding_lookup 함수로 Word Embedding 수행
with tf.name_scope('embeddings'):
    '''
        ▣ Word Embedding
         - 고차원의 원-핫 벡터를 저차원의 고밀도 벡터로 매핑하는 것.
    '''
    embeddings = tf.Variable(tf.random_uniform(shape=[vocabulary_size,
                                                      embedding_dimension],
                                               minval=-1.0, maxval=1.0), name='embedding')
    embed = tf.nn.embedding_lookup(embeddings, _inputs)  # outputs: [batch_size, time_steps, embedding_dimension]

#todo LSTM Cell 을 사용한 RNN 신경망 부
with tf.variable_scope('lstm'):
    '''
        outputs, states = tf.nn.dynamic_rnn 에서 states 는 outputs 에서 가장 마지막 출력 벡터를 나타낸다.
        sequence_length 옵션으로 가변 길이 시퀀스에 대해 학습을 할 수 있다.
    '''
    #todo Single-layer LSTM
    # lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size, forget_bias=1.0)
    # outputs, states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=embed, sequence_length=_seqlens, dtype=tf.float32)

    #todo Multi-layer LSTM
    num_LSTM_layers = 2
    lstm_cell_list = [tf.contrib.rnn.BasicLSTMCell(hidden_layer_size, forget_bias=1.0) for _ in range(num_LSTM_layers)]
    cell = tf.contrib.rnn.MultiRNNCell(cells=lstm_cell_list, state_is_tuple=True)
    outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=embed, sequence_length=_seqlens, dtype=tf.float32)

weights = {'linear_layer': tf.Variable(tf.truncated_normal(shape=[hidden_layer_size, num_classes],
                                                           mean=0, stddev=.01))}
biases = {'linear_layer': tf.Variable(tf.truncated_normal(shape=[num_classes],
                                                          mean=0, stddev=.01))}

# 최종 상태를 뽑아 선형 계층에 적용
#todo Single-layer LSTM
# final_output = tf.matmul(states[1], weights['linear_layer']) + biases['linear_layer']
#todo Multi-layer LSTM
final_output = tf.matmul(states[num_LSTM_layers-1][1], weights['linear_layer']) + biases['linear_layer']

softmax = tf.nn.softmax_cross_entropy_with_logits(logits=final_output, labels=_labels)
cross_entropy = tf.reduce_mean(softmax)

#todo 학습 부
train_step = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.9).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(final_output, 1), tf.argmax(_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1000):
        x_batch, y_batch, seqlen_batch = get_sentence_batch(batch_size, train_x, train_y, train_seqlens)
        sess.run(train_step, feed_dict={_inputs: x_batch, _labels: y_batch, _seqlens: seqlen_batch})

        if step % 100 == 0:
            acc = sess.run(accuracy, feed_dict={_inputs: x_batch, _labels: y_batch, _seqlens: seqlen_batch})
            print('Accuracy at %d: %.5f' % (step, acc))

    for test_batch in range(5):
        x_test, y_test, seqlen_test = get_sentence_batch(batch_size, test_x, test_y, test_seqlens)
        test_acc = sess.run(accuracy, feed_dict={_inputs: x_test, _labels: y_test, _seqlens: seqlen_test})
        print('Test batch accuracy %d: %.5f' % (test_batch, test_acc))

    output_example = sess.run(outputs, feed_dict={_inputs: x_test, _labels: y_test, _seqlens: seqlen_test})
    # todo Single-layer LSTM
    # states_example = sess.run(states[1], feed_dict={_inputs: x_test, _labels: y_test, _seqlens: seqlen_test})
    # todo Multi-layer LSTM
    states_example = sess.run(states[num_LSTM_layers-1][1], feed_dict={_inputs: x_test, _labels: y_test, _seqlens: seqlen_test})

    print(seqlen_test[1])
    print(output_example.shape, states_example.shape)
    print(output_example[1].shape)
    print(output_example[1][:6, 0:3])
    print(states_example[1][0:3])