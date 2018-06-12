import numpy as np
import tensorflow as tf

#todo 텐서 배열과 형태
c = tf.constant([[1, 2, 3],
                 [4, 5, 6]])
print("Python List input: {}".format(c.get_shape()))

c = tf.constant(np.array(
    [[[1, 2, 3],
      [4, 5, 6]],
     [[1, 1, 1],
      [2, 2, 2]]
     ]))
print("3d NumPy array input: {}".format(c.get_shape()))

sess = tf.InteractiveSession()
c = tf.linspace(0.0, 4.0, 5)
print("The content of 'c': \n {}\n".format(c.eval()))
sess.close()

#todo 행렬곱
A = tf.constant([[1, 2, 3],
                 [4, 5, 6]])
print(A.get_shape())

x = tf.constant([1, 0, 1])
print(x.get_shape())

"""Tensorflow 에서는 broadcasting 기능이 지원되지 않아서 차원 확장을 해서 연산을 수행해야 한다."""
x = tf.expand_dims(input=x, axis=1)
print(x.get_shape())

b = tf.matmul(A, x)

sess = tf.InteractiveSession()
print('matmul result:\n {}'.format(b.eval()))
sess.close()

#todo 이름
with tf.Graph().as_default():
    c1 = tf.constant(4, dtype=tf.float64, name='c')
    c2 = tf.constant(4, dtype=tf.int32, name='c')

print(c1.name)
print(c2.name)

#todo 이름 스코프
with tf.Graph().as_default():
    c1 = tf.constant(4, dtype=tf.float64, name='c')
    with tf.name_scope('prefix_name'):
        """
           ▣ tf.name_scope vs tf.variable_scope
            - tf.name_scope 내에 선언된 tf.get_variable() 변수는 scope 의 영향을 받지 않지만, tf.variable_scope 내에 선언된 
              tf.get_variable() 변수는 scope 의 영향을 받는다.
        """
        c2 = tf.constant(4, dtype=tf.int32, name='c')
        c3 = tf.constant(4, dtype=tf.float64, name='c')

print(c1.name)
print(c2.name)
print(c3.name)