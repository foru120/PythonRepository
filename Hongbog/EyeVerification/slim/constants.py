import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string('train_right_data_dir',
                    'G:\\04_dataset\\tfrecords\\eye\\train\\right',
                    '오른쪽 눈 학습 데이터 경로')

flags.DEFINE_string('train_left_data_dir',
                    'G:\\04_dataset\\tfrecords\\eye\\train\\left',
                    '왼쪽 눈 학습 데이터 경로')

flags.DEFINE_string('test_right_data_dir',
                    'G:\\04_dataset\\tfrecords\\eye\\test\\right',
                    '오른쪽 눈 테스트 데이터 경로')

flags.DEFINE_string('test_left_data_dir',
                    'G:\\04_dataset\\tfrecords\\eye\\test\\left',
                    '왼쪽 눈 테스트 데이터 경로')

flags.DEFINE_integer('batch_size',
                     50,
                     '훈련 시 배치 단위')

flags.DEFINE_float('learning_rate',
                   0.001,
                   '훈련 시 Learning Rate')