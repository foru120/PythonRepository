import tensorflow as tf
import numpy as np

# Rank  : 차원의 개수
# Shape : 각 차원당 원소의 개수
# Axis  : 가장 바깥 차원이 0, 안쪽 차원일수록 숫자가 늘어난다.
t = tf.constant([1, 2, 3, 4])
sess = tf.Session()
tf.shape(t).eval(session=sess)

t = tf.constant([[1, 2], [3, 4]])
tf.shape(t).eval(session=sess)

t = tf.constant([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
tf.shape(t).eval(session=sess)

# Matmul, multiply
matrix1 = tf.constant([[1., 2.], [3., 4.]])
matrix2 = tf.constant([[1.], [2.]])
print('matrix 1 shape', matrix1.shape)
print('matrix 2 shape', matrix2.shape)
tf.matmul(matrix1, matrix2).eval(session=sess)

# Broadcasting
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2., 2.]])
(matrix1 + matrix2).eval(session=sess)

matrix1 = tf.constant([[1., 2.]])
matrix2 = tf.constant(3.)
(matrix1 + matrix2).eval(session=sess)

# Reduce_mean, Reduce_sum
tf.reduce_mean([1, 2], axis=0).eval(session=sess)  # 평균을 구할땐 float 으로 변환해주어야 한다

x = [[1., 2.], [3., 4.]]
tf.reduce_mean(x).eval(session=sess)  # axis 를 설정하지 않으면 전체 데이터에 대한 mean 을 구한다.
tf.reduce_mean(x, axis=0).eval(session=sess)
tf.reduce_mean(x, axis=1).eval(session=sess)
tf.reduce_mean(x, axis=-1).eval(session=sess)

tf.reduce_sum(x).eval(session=sess)
tf.reduce_sum(x, axis=0).eval(session=sess)
tf.reduce_sum(x, axis=1).eval(session=sess)

# Argmax
x = [[0, 1, 2], [2, 1, 0]]
tf.argmax(x, axis=0).eval(session=sess)
tf.argmax(x, axis=1).eval(session=sess)
tf.argmax(x, axis=-1).eval(session=sess)

# Reshape
t = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])
t.shape  # (2, 2, 3)
tf.reshape(t, shape=[-1, 3]).eval(session=sess)  # (4, 3)
tf.reshape(t, shape=[-1, 1, 3]).eval(session=sess)  # (4, 1, 3)

# Squeeze, Expand
tf.squeeze([[0], [1], [2]]).eval(session=sess)  # 차원 감소
tf.expand_dims([0, 1, 2], 1).eval(session=sess)  # 차원 확대

# One hot
t = tf.one_hot([[0], [1], [2], [0]], depth=3).eval(session=sess)  # one hot 시 차원이 하나 추가된다.
tf.reshape(t, shape=[-1, 3]).eval(session=sess)  # 원래 형태로 되돌린다.

# Stack
x = [1, 4]
y = [2, 5]
z = [3, 6]
tf.stack([x, y, z]).eval(session=sess)
tf.stack([x, y, z], axis=1).eval(session=sess)

# Ones and Zeros like
x = [[0, 1, 2], [2, 1, 0]]
tf.ones_like(x).eval(session=sess)
tf.zeros_like(x).eval(session=sess)