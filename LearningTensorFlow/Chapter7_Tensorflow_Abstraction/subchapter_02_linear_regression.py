from sklearn import datasets, metrics, preprocessing
import tensorflow as tf

#todo tensorflow 소스로 선형회귀모델 구현
boston = datasets.load_boston()
x_data = preprocessing.StandardScaler().fit_transform(boston.data)
y_data = boston.target

x = tf.placeholder(dtype=tf.float64, shape=(None, 13))
y_true = tf.placeholder(dtype=tf.float64, shape=(None))

with tf.name_scope('inference') as scope:
    w = tf.Variable(initial_value=tf.zeros(shape=[1, 13], dtype=tf.float64, name='weights'))
    b = tf.Variable(initial_value=0, dtype=tf.float64, name='bias')
    y_pred = tf.matmul(w, tf.transpose(x)) + b  # 입력에 대해 transpose 한 경우와 w 를 transpose 한 경우 loss 값이 차이가 많이 발생 (입력에 대한 transpose 시 더 적은 loss 발생)

with tf.name_scope('loss') as scope:
    loss = tf.reduce_mean(tf.square(y_true-y_pred))

with tf.name_scope('train') as scope:
    learning_rate = 0.1
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(200):
        sess.run(train, {x: x_data, y_true: y_data})

    MSE = sess.run(loss, feed_dict={x: x_data, y_true: y_data})

print(MSE)

#todo contrib.learn 을 사용해 선형회귀모델 구현
import tensorflow.contrib.learn as learn

boston = datasets.load_boston()
x_data = preprocessing.StandardScaler().fit_transform(boston.data)
y_data = boston.target

NUM_STEPS = 200
MINIBATCH_SIZE = 506

feature_columns = learn.infer_real_valued_columns_from_input(x_data)

reg = learn.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1)
)
reg.fit(x=x_data, y=y_data, steps=NUM_STEPS, batch_size=MINIBATCH_SIZE)
MSE = reg.evaluate(x_data, y_data, steps=1)
print(MSE)