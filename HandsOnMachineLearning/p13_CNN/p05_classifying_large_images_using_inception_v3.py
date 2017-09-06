from HandsOnMachineLearning.p13_CNN.setup import *

# 8.a. Exercise: Download some images of various animals. Load them in Python, for example using the matplotlib.image.mpimg.imread() function.
#                Resize and/or crop them to 299 × 299 pixels, and ensure that they have just three channels (RGB), with no transparency channel.
width = 299
height = 299
channels = 3

import matplotlib.image as mpimg
import os
test_image = mpimg.imread(os.path.join('images', 'cnn', 'test_image.png'))[:, :, :channels]
plt.imshow(test_image)
plt.axis('off')
plt.show()

# 8.b. Exercise: Download the latest pretrained Inception v3 model: the checkpoint is available at https://goo.gl/nxSQvl[].
import sys
import tarfile
from six.moves import urllib

TF_MODELS_URL = 'http://download.tensorflow.org/models'
INCEPTION_V3_URL = TF_MODELS_URL + '/inception_v3_2016_08_28.tar.gz'
INCEPTION_PATH = os.path.join('datasets', 'inception')
INCEPTION_V3_CHECKPOINT_PATH = os.path.join(INCEPTION_PATH, 'inception_v3.ckpt')

def download_progress(count, block_size, total_size):
    percent = count * block_size * 100 // total_size
    sys.stdout.write('\rDownloading: {}%'.format(percent))
    sys.stdout.flush()

# inception_v3 tgz 파일을 다운받아서 압축을 해제하는 함수
def fetch_pretrained_inception_v3(url=INCEPTION_V3_URL, path=INCEPTION_PATH):
    if os.path.exists(INCEPTION_V3_CHECKPOINT_PATH):
        return
    os.makedirs(path, exist_ok=True)
    tgz_path = os.path.join(path, 'inception_v3.tgz')
    urllib.request.urlretrieve(url, tgz_path, reporthook=download_progress)
    inception_tgz = tarfile.open(tgz_path)
    inception_tgz.extractall(path=path)
    inception_tgz.close()
    os.remove(tgz_path)

fetch_pretrained_inception_v3()

import re

CLASS_NAME_REGEX = re.compile(r'^n\d+\s+(.*)\s*$', re.M | re.U)

def load_class_names():
    with open(os.path.join('datasets', 'inception', 'imagenet_class_names.txt'), 'rb') as f:
        content = f.read().decode('utf-8')
        return CLASS_NAME_REGEX.findall(content)

class_names = load_class_names()
print(class_names[:5])

# 8.c. Exercise: Create the Inception v3 model by calling the inception_v3() function, as shown below.
#                This must be done within an argument scope created by the inception_v3_arg_scope() function.
#                Also, you must set is_training=False and num_classes=1001
from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim

reset_graph()

X = tf.placeholder(tf.float32, shape=[None, 299, 299, 3], name='X')
with slim.arg_scope(inception.inception_v3_arg_scope()):
    logits, end_points = inception.inception_v3(X, num_classes=1001, is_training=False)
predictions = end_points['Predictions']
saver = tf.train.Saver()

# 8.d. Exercise: Open a session and use the Saver to restore the pretrained model checkpoint you downloaded earlier.
with tf.Session() as sess:
    saver.restore(sess, INCEPTION_V3_CHECKPOINT_PATH)

# 8.e. Exersice: Run the model to classify the images you prepared.
#      Display the top five predictions for each image, along with the estimated probability
#     (the list of class names is available at https://goo.gl/brXRtZ[]). How accurate is the model?
X_test = test_image.reshape(-1, height, width, channels)

with tf.Session() as sess:
    saver.restore(sess, INCEPTION_V3_CHECKPOINT_PATH)
    predictions_val = predictions.eval(feed_dict={X: X_test})

most_likely_class_index = np.argmax(predictions_val[0])
print(most_likely_class_index)
print(class_names[most_likely_class_index])

# np.argpartition : 특정 kth 인덱스에 해당 하는 값을 중심으로 partitioning 수행하고,  작은 수는 왼쪽 큰 수는 오른쪽으로 배치한다.
#                   최종 출력 값은 인덱스가 출력된다.
# np.argsort : 간접 정렬을 수행. (정렬된 인덱스 순서가 출력)
top_5 = np.argpartition(predictions_val[0], -5)[-5:]
top_5 = top_5[np.argsort(predictions_val[0][top_5])]
for i in top_5:
    print('{0}: {1: .2f}%'.format(class_names[i], 100 * predictions_val[0][i]))