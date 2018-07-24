import os
# import scipy
import numpy as np
import tensorflow as tf

from DeepLearningTechniques.CapsNet.config import cfg

def load_mnist(path, is_training):
    # train dataset
    fd = open(os.path.join(cfg.dataset, 'train-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)  # text or binary file 안에 있는 데이터를 배열로 구성하는 함수
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)  # 0~15 번째 인덱스 까지 불필요한 데이터가 들어가있어 제거

    fd = open(os.path.join(cfg.dataset, 'train-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)  # 0~7 번째 인덱스 까지 불필요한 데이터가 들어가있어 제거

    # test dataset
    fd = open(os.path.join(cfg.dataset, 't10k-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(cfg.dataset, 't10k-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    # normalization and convert to a tensor [60000, 28, 28, 1]
    trX = tf.convert_to_tensor(trX / 255., tf.float32)  # numpy.ndarray -> tensor
    teX = tf.convert_to_tensor(teX / 255., tf.float32)  # numpy.ndarray -> tensor

    # => [num_samples, 10]
    trY = tf.one_hot(trY, depth=10, axis=1, dtype=tf.float32)  # return to tensor
    teY = tf.one_hot(teY, depth=10, axis=1, dtype=tf.float32)

    if is_training:
        return trX, trY
    else:
        return teX, teY

def get_batch_data():
    trX, trY = load_mnist(cfg.dataset, cfg.is_training)  # mnist 전체 데이터 로드하는 함수

    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues,
                                  num_threads=cfg.num_threads,
                                  batch_size=cfg.batch_size,
                                  capacity=cfg.batch_size * 64,
                                  min_after_dequeue=cfg.batch_size * 32,
                                  allow_smaller_final_batch=False)  # queue 에 남아있는 아이템이 충분하지 않으면 마지막 batch 를 작게 만드는 옵션
    return (X, Y)

# def save_images(imgs, size, path):
#     '''
#     Args:
#         imgs: [batch_size, image_height, image_width]
#         size: a list with tow int elements, [image_height, image_width]
#         path: the path to save images
#     '''
#     imgs = (imgs + 1.) / 2  # inverse_transform
#     return(scipy.misc.imsave(path, mergeImgs(imgs, size)))

def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs

if __name__ == '__main__':
    X, Y = load_mnist(cfg.dataset, cfg.is_training)
    print(X.get_shape())
    print(X.dtype)