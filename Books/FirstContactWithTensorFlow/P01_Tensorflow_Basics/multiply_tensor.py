import tensorflow as tf

a = tf.placeholder('float')  # placeholder : 프로그램 실행 중에 값을 변경할 수 있는 '심벌릭' 변수
b = tf.placeholder('float')

y = tf.multiply(a, b)

sess = tf.Session()

print(sess.run(y, feed_dict={a: 3, b: 3}))