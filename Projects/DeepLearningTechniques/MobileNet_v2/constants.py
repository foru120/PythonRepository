import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('data_path', '/home/kyh/dataset/gelontoxon_v1', '데이터 셋 저장 경로')
flags.DEFINE_string('trained_param_path',
                    '/home/kyh/PycharmProjects/PythonRepository/Projects/DeepLearningTechniques/MobileNet_v2/train_log/3th_test',
                    '훈련된 파라미터 값 저장 경로')
flags.DEFINE_string('deploy_log_dir',
                    '/home/kyh/PycharmProjects/PythonRepository/Projects/DeepLearningTechniques/MobileNet_v2/deploy_log/3th_test',
                    'Model Deploy 시 사용될 체크포인트 파일 저장 경로')

flags.DEFINE_integer('epochs', 200, '훈련 시 에폭 수')
flags.DEFINE_integer('batch_size', 10, '훈련 시 배치 크기')

flags.DEFINE_float('dropout_rate', 0.6, '훈련 시 드롭아웃 비율')
flags.DEFINE_float('learning_rate', 0.045, '훈련 시 학습 률')
flags.DEFINE_integer('decay_step', 6, '학습 률 감소 단계')
flags.DEFINE_float('decay_rate', 0.98, '학습 률 감소 률')