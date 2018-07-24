import numpy as np
import tensorflow as tf

points = np.zeros((2000, 2))
vectors = tf.constant(points)
expanded_vectors = tf.expand_dims(vectors, 0)

print(points)
print(vectors)
print(expanded_vectors, expanded_vectors.get_shape())