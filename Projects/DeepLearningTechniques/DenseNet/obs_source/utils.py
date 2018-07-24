import tensorflow as tf
import numpy as np
import cv2

import tensorflow.contrib.slim as slim

def save_image(images, dir):
	batch_size = np.shape(images)[0]
	size = min(batch_size, 100)
	size = int(np.floor(np.sqrt(size)))
	
	img = np.zeros(shape=[28 * size, 28 * size])
	
	for i in range(size):
		for j in range(size):
			TEMP = images[i * size + j, :, :, 0]
			minval = np.amin(TEMP)
			maxval = np.amax(TEMP)
			
			TEMP = (TEMP - minval) / (maxval - minval) * 255.
			
			img[i * 28:i * 28 + 28, j * 28:j * 28 + 28] = TEMP
	
	cv2.imwrite(dir, img)


def dense_to_one_hot(label_batch, num_classes):
	one_hot = tf.map_fn(lambda x : tf.cast(slim.one_hot_encoding(x, num_classes), tf.int32), label_batch)
	one_hot = tf.reshape(one_hot, [-1, num_classes])
	return one_hot
