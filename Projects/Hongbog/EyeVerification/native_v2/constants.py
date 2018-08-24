import tensorflow as tf

flags = tf.app.flags

'''학습 데이터 경로'''
flags.DEFINE_string('right_train_data_path',
                    'G:/04_dataset/eye_verification/eye_dataset_v2/train/right',
                    '오른쪽 눈 학습 데이터 경로')

flags.DEFINE_string('right_test_data_path',
                    'G:/04_dataset/eye_verification/eye_dataset_v2/train/right',
                    '오른쪽 눈 테스트 데이터 경로')

flags.DEFINE_string('left_train_data_path',
                    'G:/04_dataset/eye_verification/eye_dataset_v2/train/left',
                    '왼쪽 눈 학습 데이터 경로')

flags.DEFINE_string('left_test_data_path',
                    'G:/04_dataset/eye_verification/eye_dataset_v2/train/left',
                    '왼쪽 눈 테스트 데이터 경로')

'''학습 로그 경로'''
flags.DEFINE_string('trained_weight_dir',
                    'D:/Source/PythonRepository/Projects/Hongbog/EyeVerification/native_v2/train_log/008',
                    '훈련된 가중치 값 저장 경로')

flags.DEFINE_string('saved_weight_dir',
                    'D:/Source/PythonRepository/Projects/Hongbog/EyeVerification/native_v2/train_log/009',
                    '훈련된 가중치 값 저장 경로')

flags.DEFINE_string('tensorboard_log_dir',
                    'D:/Source/PythonRepository/Projects/Hongbog/EyeVerification/native_v2/tensorboard_log/009',
                    '텐서보드에서 모니터링 변수 저장 경로')

flags.DEFINE_string('deploy_log_dir',
                    'D:/Source/PythonRepository/Projects/Hongbog/EyeVerification/native_v2/deploy_log/009',
                    'Model Deploy 시 사용될 체크포인트 파일 저장 경로')

'''하이퍼 파라미터'''
flags.DEFINE_integer('epochs',
                     150,
                     '훈련 시 총 에폭 수')

flags.DEFINE_integer('batch_size',
                     50,
                     '훈련 시 배치 크기')

flags.DEFINE_float('dropout_rate',
                   0.4,
                   '신경망 dropout 비율')

flags.DEFINE_float('learning_rate',
                   0.001,
                   '신경망 learning 비율')

flags.DEFINE_float('regularization_scale',
                   0.0005,
                   '신경망 L2 regularization 크기')