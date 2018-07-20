import tensorflow as tf

sess = tf.InteractiveSession()
queue1 = tf.FIFOQueue(capacity=10, dtypes=[tf.string])

#todo Queue Data 삽입
enque_op = queue1.enqueue(['F'])
print(sess.run(queue1.size()))

enque_op.run()
print(sess.run(queue1.size()))

enque_op = queue1.enqueue(['I'])
enque_op.run()
enque_op = queue1.enqueue(['F'])
enque_op.run()
enque_op = queue1.enqueue(['O'])
enque_op.run()

print(sess.run(queue1.size()))

#todo Queue Data 추출
x = queue1.dequeue()
print(x.eval())
print(x.eval())
print(x.eval())
print(x.eval())

queue1 = tf.FIFOQueue(capacity=10, dtypes=[tf.string], shapes=[()])
enque_op = queue1.enqueue(['F'])
enque_op.run()
enque_op = queue1.enqueue(['I'])
enque_op.run()
enque_op = queue1.enqueue(['F'])
enque_op.run()
enque_op = queue1.enqueue(['O'])
enque_op.run()

inputs = queue1.dequeue_many(4)
print(inputs.eval())

#todo Multi-Threading
import threading
import time

gen_random_normal = tf.random_normal(shape=())
queue = tf.FIFOQueue(capacity=100, dtypes=[tf.float32], shapes=())
enque = queue.enqueue(gen_random_normal)

def add():
    for i in range(10):
        sess.run(enque)

threads = [threading.Thread(target=add, args=()) for i in range(10)]
print(threads)

for t in threads:
    t.start()

print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))

x = queue.dequeue_many(10)
print(x.eval())
print(sess.run(queue.size()))

sess.close()