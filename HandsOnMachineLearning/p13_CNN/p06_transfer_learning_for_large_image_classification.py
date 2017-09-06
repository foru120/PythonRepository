# 9.a. Exercise: Create a training set containing at least 100 images per class.
#                For example, you could classify your own pictures based on the location (beach, mountain, city, etc.),
#                or alternatively you can just use an existing dataset, such as the flowers dataset or MIT's places dataset
#               (requires registration, and it is huge).
import sys
import tarfile
from six.moves import urllib
import os
import numpy as np
from HandsOnMachineLearning.p13_CNN.setup import *

FLOWERS_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'
FLOWERS_PATH = os.path.join('datasets', 'flowers')

def download_progress(count, block_size, total_size):
    percent = count * block_size * 100 // total_size
    sys.stdout.write('\rDownloading: {}%'.format(percent))
    sys.stdout.flush()

def fetch_flowers(url=FLOWERS_URL, path=FLOWERS_PATH):
    if os.path.exists(FLOWERS_PATH):
        return
    os.makedirs(path, exist_ok=True)
    tgz_path = os.path.join(path, 'flower_photos.tgz')
    urllib.request.urlretrieve(url, tgz_path, reporthook=download_progress)
    flowers_tgz = tarfile.open(tgz_path)
    flowers_tgz.extractall(path=path)  # 압축 파일의 전체 내용을 특정 path 에 풀어주는 함수
    flowers_tgz.close()
    os.remove(tgz_path)

fetch_flowers()

flowers_root_path = os.path.join(FLOWERS_PATH, 'flower_photos')
flower_classes = sorted([dirname for dirname in os.listdir(flowers_root_path) if os.path.isdir(os.path.join(flowers_root_path, dirname))])

from collections import defaultdict

image_paths = defaultdict(list)

for flower_class in flower_classes:
    image_dir = os.path.join(flowers_root_path, flower_class)
    for filepath in os.listdir(image_dir):
        if filepath.endswith('.jpg'):
            image_paths[flower_class].append(os.path.join(image_dir, filepath))

for paths in image_paths.values():
    paths.sort()

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

n_examples_per_class = 2

for flower_class in flower_classes:
    print('Class:', flower_class)
    plt.figure(figsize=(10, 5))
    for index, example_image_path in enumerate(image_paths[flower_class][:n_examples_per_class]):
        example_image = mpimg.imread(example_image_path)[:, :, :3]  # [H, W, C]
        plt.subplot(100 + n_examples_per_class * 10 + index + 1)
        plt.title('{}x{}'.format(example_image.shape[1], example_image.shape[0]))
        plt.imshow(example_image)
        plt.axis('off')
    plt.show()

# 9.b. Exercise: Write a preprocessing step that will resize and crop the image to 299 × 299, with some randomness for data augmentation.
# First, let's implement this using NumPy and SciPy:
# - using basic NumPy slicing for image cropping,
# - NumPy's fliplr() function to flip the image horizontally (with 50% probability),
# - and SciPy's imresize() function for zooming.
# - Note that imresize() is based on the Python Image Library (PIL).
from scipy.misc import imresize

# 이미지 전처리 하는 함수
def prepare_image_with_numpy(image, target_width=299, target_height=299, max_zoom=0.2):
    '''Zooms and crops the image randomly for data augmentation.'''
    height = image.shape[0]
    width = image.shape[1]
    image_ratio = width / height
    target_image_ratio = target_width / target_height
    crop_vertically = image_ratio < target_image_ratio  # image 의 height 가 width 보다 크면 False 작으면 True
    crop_width = width if crop_vertically else int(height * target_image_ratio)
    crop_height = int(width / target_image_ratio) if crop_vertically else height

    # Now let's shrink this bounding box by a random factor (dividing the dimensions by a random number)
    # between 1.0 and 1.0 + 'max_zoom'
    resize_factor = np.random.rand() * max_zoom + 1.0
    crop_width = int(crop_width / resize_factor)
    crop_height = int(crop_height / resize_factor)

    # Next, we can select a random location on the image for this bounding box.
    x0 = np.random.randint(0, width - crop_width)
    y0 = np.random.randint(0, height - crop_height)
    x1 = x0 + crop_width
    y1 = y0 + crop_height

    # Let's crop the image using the random bounding box we built.
    image = image[y0:y1, x0:x1]

    # Let's also flip the image horizontally with 50% probability.
    if np.random.rand() < 0.5:
        image = np.fliplr(image)  # 이미지를 수평으로 뒤집는 효과

    # Now, let's resize the image to the target dimensions.
    image = imresize(image, (target_width, target_height))

    # Finally, the Convolution Neural Network expects colors represented as
    # 32-bit floats ranging from 0.0 to 1.0:
    return image.astype(np.float32) / 255

plt.figure(figsize=(6, 8))
plt.imshow(example_image)
plt.title('{}x{}'.format(example_image.shape[1], example_image.shape[0]))
plt.axis('off')
plt.show()

prepare_image = prepare_image_with_numpy(example_image)

plt.figure(figsize=(8, 8))
plt.imshow(prepare_image)
plt.title('{}x{}'.format(prepare_image.shape[1], prepare_image.shape[0]))
plt.axis('off')
plt.show()

# Now let's look at a few other random images generated from the same original image
rows, cols = 2, 3

plt.figure(figsize=(14, 8))
for row in range(rows):
    for col in range(cols):
        prepare_image = prepare_image_with_numpy(example_image)
        plt.subplot(rows, cols, row * cols + col + 1)
        plt.title('{}x{}'.format(prepare_image.shape[1], prepare_image.shape[0]))
        plt.imshow(prepare_image)
        plt.axis('off')
plt.show()

# tensoflow 를 이용한 이미지 전처리
import tensorflow as tf
def prepare_image_with_tensorflow(image, target_width=299, target_height=299, max_zoom=0.2):
    image_shape = tf.cast(tf.shape(image), tf.float32)
    height = image_shape[0]
    width = image_shape[1]
    image_ratio = width / height
    target_image_ratio = target_width / target_height
    crop_vertically = image_ratio < target_image_ratio
    crop_width = tf.cond(crop_vertically,
                         lambda: width,
                         lambda: height * target_image_ratio)  # tf.cond(조건, True 인 경우 함수 호출, False 인 경우 함수 호출)
    crop_height = tf.cond(crop_vertically,
                          lambda: width / target_image_ratio,
                          lambda: height)

    resize_factor = tf.random_uniform(shape=[], minval=1.0, maxval=1.0 + max_zoom)
    crop_width = tf.cast(crop_width / resize_factor, tf.int32)
    crop_height = tf.cast(crop_height / resize_factor, tf.int32)
    box_size = tf.stack([crop_height, crop_width, 3])

    image = tf.random_crop(image, box_size)
    image = tf.image.random_flip_left_right(image)
    image_batch = tf.expand_dims(image, 0)

    image_batch = tf.image.resize_bilinear(image_batch, [target_height, target_width])
    image = image_batch[0] / 255
    return image

reset_graph()

input_image = tf.placeholder(tf.unit8, shape=[None, None, 3])
prepared_image_op = prepare_image_with_tensorflow(input_image)

with tf.Session():
    prepared_image = prepared_image_op.eval(feed_dict={input_image: example_image})

plt.figure(figsize=(6, 6))
plt.imshow(prepared_image)

# 9.c. Exercise: Using the pretrained Inception v3 model from the previous exercise, freeze all layers up to the bottleneck layer
#                (i.e., the last layer before the output layer), and replace the output layer with the appropriate number of outputs for your new classification task
#                (e.g., the flowers dataset has five mutually exclusive classes so the output layer must have five neurons and use the softmax activation function).
from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim

reset_graph()
height, width, channels = 299, 299, 3
X = tf.placeholder(tf.float32, shape=[None, height, width, channels], name='X')
training = tf.placeholder_with_default(False, shape=[])
with slim.arg_scope(inception.inception_v3_arg_scope()):
    logits, end_points = inception.inception_v3(X, num_classes=1001, is_training=training)

inception_saver = tf.train.Saver()

prelogits = tf.squeeze(end_points['PreLogits'], axis=[1, 2])

n_outputs = len(flower_classes)

with tf.name_scope('new_output_layer'):
    flower_logits = tf.layers.dense(prelogits, n_outputs, name='flower_logits')
    Y_proba = tf.nn.softmax(flower_logits, name='Y_proba')

y = tf.placeholder(tf.int32, shape=[None])

with tf.name_scope('train'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=flower_logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    flower_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='flower_logits')
    training_op = optimizer.minimize(loss, var_list=flower_vars)

with tf.name_scope('eval'):
    correct=  tf.nn.in_top_k(flower_logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope('init_and_save'):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()