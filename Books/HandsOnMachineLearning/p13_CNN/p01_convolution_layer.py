from sklearn.datasets import load_sample_image
from HandsOnMachineLearning.p13_CNN.setup import *

china = load_sample_image('china.jpg')
flower = load_sample_image('flower.jpg')
image = china[150:220, 130:250]
height, width, channels = image.shape
image_grayscale = image.mean(axis=2).astype(np.float32)
images = image_grayscale.reshape(1, height, width, 1)

fmap = np.zeros(shape=(7, 7, 1, 2), dtype=np.float32)
fmap[:, 3, 0, 0] = 1
fmap[3, :, 0, 1] = 1
plot_image(fmap[:, :, 0, 0])  # vertical filter
plt.show()
plot_image(fmap[:, :, 0, 1])  # horizontal filter
plt.show()

reset_graph()

X = tf.placeholder(tf.float32, shape=(None, height, width, 1))
feature_maps = tf.constant(fmap)
convolution = tf.nn.conv2d(X, feature_maps, strides=[1, 2, 2, 1], padding='SAME', use_cudnn_on_gpu=False)  # with zero padding and a stride of 2

with tf.Session() as sess:
    output = convolution.eval(feed_dict={X: images})

plot_image(images[0, :, :, 0])
save_fig('china_original', tight_layout=False)
plt.show()

plot_image(output[0, :, :, 0])
save_fig("china_vertical", tight_layout=False)
plt.show()

plot_image(output[0, :, :, 1])
save_fig("china_horizontal", tight_layout=False)
plt.show()