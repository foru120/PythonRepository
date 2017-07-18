import tensorflow as tf

#  ▣ 어느 디바이스에서 처리되었는지 확인
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)

# config=tf.ConfigProto(log_device_placement=True) : 연산 및 텐서가 어느 디바이스에서 처리되었는지 확인하기 위해 옵션 추가
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))


#  ▣ 특정 디바이스에서 연산 수행하도록 설정
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))


#  ▣ 여러 GPU 에서의 병렬처리
c = []

for d in ['/gpu:2', '/gpu:3']:
    with tf.device(d):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c.append(tf.matmul(a, b))

with tf.device('/cpu:0'):
    sum = tf.add_n(c)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(sum))

#  ▣ GPU 코드 예제
#   - GPU & CPU 처리 코드
import numpy as np
import tensorflow as tf
import datetime

A = np.random.rand(10000, 10000).astype('float32')
B = np.random.rand(10000, 10000).astype('float32')

n = 10

c1 = []
c2 = []

def matpow(M, n):
    if n < 1:
        return M
    else:
        return tf.matmul(M, matpow(M, n-1))

with tf.device('/gpu:0'):
    a = tf.constant(A)
    b = tf.constant(B)
    c1.append(matpow(a, n))
    c2.append(matpow(b, n))

with tf.device('/cpu:0'):
    sum = tf.add_n(c1)

t1_1 = datetime.datetime.now()

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(sum)

t2_1 = datetime.datetime.now()

print('Single GPU computation time: ' + str(t2_1 - t1_1))

#   - GPU 2 처리 코드
import numpy as np
import tensorflow as tf
import datetime

A = np.random.rand(1e4, 1e4).astype('float32')
B = np.random.rand(1e4, 1e4).astype('float32')

n = 10

c1 = []
c2 = []

def matpow(M, n):
    if n < 1:
        return M
    else:
        return tf.matmul(M, matpow(M, n - 1))

with tf.device('/gpu:0'):
    a = tf.constant(A)
    c1.append(matpow(a, n))

with tf.device('/gpu:0'):
    b = tf.constant(B)
    c2.append(matpow(b, n))

with tf.device('/cpu:0'):
    sum = tf.add_n(c1)

t1_1 = datetime.datetime.now()

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(sum)

t2_1 = datetime.datetime.now()

print('Multi GPU computation time: ' + str(t2_1 - t1_1))