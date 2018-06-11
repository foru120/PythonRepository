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

flags.DEFINE_float('dropout_rate',
                   0.6,
                   '훈련 시 Dropout Rate')

flags.DEFINE_integer('hidden_num',
                     30,
                     'Residual Block Default FeatureMap 개수')

flags.DEFINE_float('regularizer_scale',
                   0.0005,
                   'L2 Regularizer Scale')

# from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
# print_tensors_in_checkpoint_file(file_name='D:\\Source\\PythonRepository\\Hongbog\\EyeVerification\\slim\\train_log\\002\\right\\model.ckpt-3080', tensor_name='', all_tensors=True)