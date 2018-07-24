import tensorflow as tf

init_val = tf.random_normal(shape=(1, 5), mean=0, stddev=1)
# var = tf.Variable(init_val, name='var')
var = tf.get_variable(name='var', shape=(1, 5), initializer=tf.random_normal_initializer(mean=0, stddev=1))
with tf.variable_scope(name_or_scope='', reuse=True):
    """tf.variable_scope 를 사용하면 변수를 공유할 수 있다."""
    var_1 = tf.get_variable(name='var')
print('pre run: \n{}'.format(var))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    post_var, post_var_1 = sess.run([var, var_1])

print('\npost run: \n{}'.format(post_var))
print('\npost run: \n{}'.format(post_var_1))