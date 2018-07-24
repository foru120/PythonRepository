import tensorflow as tf
import threading
import time

sess = tf.InteractiveSession()

gen_random_normal = tf.random_normal(shape=())
queue = tf.FIFOQueue(capacity=100, dtypes=[tf.float32], shapes=())
enque = queue.enqueue(gen_random_normal)

def add(coord, i):
    while not coord.should_stop():
        sess.run(enque)
        if i == 11:
            coord.request_stop()  # 어떤

coord = tf.train.Coordinator()
threads = [threading.Thread(target=add, args=(coord, i)) for i in range(10)]
coord.join(threads)

for t in threads:
    t.start()

print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))

#todo tf.train.QueueRunner & tf.RandomShuffleQueue
gen_random_normal = tf.random_normal(shape=())
'''
    tf.RandomShuffleQueue: 항목을 꺼낼 때 순서가 무작위인 큐
    min_after_dequeue: 항목 꺼내기 연산을 호출한 후 큐에 남아 있을 항목의 최소 개수
'''
queue = tf.RandomShuffleQueue(capacity=100, dtypes=[tf.float32], min_after_dequeue=1)
enqueue_op = queue.enqueue(gen_random_normal)

qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)
coord = tf.train.Coordinator()
enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
coord.request_stop()
coord.join(enqueue_threads)