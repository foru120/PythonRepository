from DeepLearningTechniques.GAN.BEGAN.slim.utils import *

class DataLoader:
    def __init__(self, batch_size, train_data_path):
        self.batch_size = batch_size

        filenames = []
        for filename in os.listdir(train_data_path):
            filenames.append(os.path.join(train_data_path, filename))

        self.train_x_len = len(filenames)
        filenames = np.asarray(filenames)[np.random.permutation(self.train_x_len)]

        with tf.variable_scope('dataloader'):
            self.train_x = tf.convert_to_tensor(filenames, dtype=tf.string, name='train_x')

    def train_setter(self, x_path):
        with tf.variable_scope('train_setter'):
            # img = tf.cast(tf.image.resize_images(tf.image.decode_png(tf.read_file(x_path), channels=3, name='image'), size=(192, 256)), tf.float32)
            img = tf.cast(tf.image.resize_images(tf.image.decode_png(tf.read_file(x_path), channels=3, name='image'), size=(128, 128)), tf.float32)
            scaled_img = tf.subtract(tf.divide(img, 127.5), 1)

        return scaled_img

    def train_loader(self):
        with tf.variable_scope('train_loader'):
            # 데이터셋을 불러오는데 불러오는 데이터는 텐서타입이어서 배치단위로 계속 부르기 위해 텐서슬라이스 함수를 반복적으로 사용한다.
            dataset = tf.contrib.data.Dataset.from_tensor_slices(self.train_x).repeat()

            # 데이터셋의 맵함수를 통해 배치사이즈별로 잘라내는데 사용하는 함수를 맵함수 안에 넣어준다.
            dataset_map = dataset.map(self.train_setter).batch(self.batch_size)

            # 데이터셋을 이터레이터를 통해 지속적으로 불러준다.
            iterator = dataset_map.make_one_shot_iterator()

            # 세션이 런 될 때마다 반복해서 이터레이터를 소환한다. 그렇게 해서 다음 배치 데이터셋을 불러온다.
            batch_input = iterator.get_next()

        return batch_input

def main():
    if not FLAGS.dataset_name:
        raise ValueError('You must supply the dataset name with --dataset_name')
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')


if __name__ == '__main__':
    tf.app.run()