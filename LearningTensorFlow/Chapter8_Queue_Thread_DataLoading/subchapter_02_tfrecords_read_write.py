import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist
import numpy as np

save_dir = './Mnist_data'

# save_dir 에 데이터 내려받기
data_sets = mnist.read_data_sets(save_dir,
                                 dtype=tf.uint8,
                                 reshape=False,
                                 validation_size=1000)

data_splits = ['train', 'test', 'validation']

#todo mnist dataset -> tfrecord 변환
for d in range(len(data_splits)):
    print('saving:' + data_splits[d])
    data_set = data_sets[d]
    print('data_set.images shape:', data_set.images.shape, ', data_set.labels shape:', data_set.labels.shape)

    filename = os.path.join(save_dir, 'tfrecord', data_splits[d] + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(filename)

    for index in range(data_set.images.shape[0]):
        image = data_set.images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[data_set.images.shape[1]])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[data_set.images.shape[2]])),
            'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[data_set.images.shape[3]])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(data_set.labels[index])])),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
        }))

        writer.write(example.SerializeToString())

    writer.close()

#todo tfrecord data read
filename = os.path.join(save_dir, 'tfrecord', 'train.tfrecords')
record_iterator = tf.python_io.tf_record_iterator(filename)
serialized_img_example = next(record_iterator)

example = tf.train.Example()
example.ParseFromString(serialized_img_example)
image = example.features.feature['image_raw'].bytes_list.value
label = example.features.feature['label'].int64_list.value[0]
width = example.features.feature['width'].int64_list.value[0]
height = example.features.feature['height'].int64_list.value[0]

img_flat = np.fromstring(image[0], dtype=np.uint8)
img_reshaped = img_flat.reshape((height, width, -1))

print(img_reshaped)