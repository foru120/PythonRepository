from HandsOnMachineLearning.p14_RNN_Github.setup import *

n_steps = 2
n_inputs = 3
n_neurons = 5

reset_graph()

X = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_inputs])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

init = tf.global_variables_initializer()

X_batch = np.array([
    [[0, 1, 2], [9, 8, 7]],
    [[3, 4, 5], [0, 0, 0]],
    [[6, 7, 8], [6, 5, 4]],
    [[9, 0, 1], [3, 2, 1]]
])

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5

with tf.Session(config=config) as sess:
    init.run()
    outputs_val = outputs.eval(feed_dict={X: X_batch})

print(outputs_val)
show_graph(tf.get_default_graph())