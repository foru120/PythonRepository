import numpy as np
import tensorflow as tf

elems = np.array(['T', 'e', 'n', 's', 'o', 'r', ' ', 'F', 'l', 'o', 'w'])
scan_sum = tf.scan(lambda a, x: a + x, elems)

sess = tf.InteractiveSession()
print(sess.run(scan_sum))
sess.close()