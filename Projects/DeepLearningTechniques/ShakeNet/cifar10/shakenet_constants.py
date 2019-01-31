import tensorflow as tf

flags = tf.app.flags

#todo 학습 데이터/로그 경로
flags.DEFINE_string('data_path',
                    'G:/04_dataset/cifar10_original',
                    '학습 데이터 경로 (cifar-10)')

flags.DEFINE_string('train_log_path',
                    'D:/Source/PythonRepository/Projects/DeepLearningTechniques/ShakeNet/cifar10/train_log',
                    '훈련 시 체크포인트 파일 저장 경로')

flags.DEFINE_string('deploy_log_path',
                    'D:/Source/PythonRepository/Projects/DeepLearningTechniques/ShakeNet/cifar10/deploy_log',
                    'Model Deploy 시 사용될 체크포인트 파일 저장 경로')

flags.DEFINE_string('cam_log_path',
                    'D:/Source/PythonRepository/Projects/DeepLearningTechniques/ShakeNet/cifar10/cam_log',
                    'CAM 이미지 및 체크포인트 파일 저장 경로')

flags.DEFINE_string('tensorboard_log_path',
                    'D:/Source/PythonRepository/Projects/DeepLearningTechniques/ShakeNet/cifar10/tensorboard_log',
                    'Tensor Board 에서 시각화하기 위해 저장되는 로그 경로')

flags.DEFINE_string('roc_curve_log_path',
                    'D:/Source/PythonRepository/Projects/DeepLearningTechniques/ShakeNet/cifar10/roc_curve_log',
                    'ROC Curve 그래프가 저장 되는 경로')

flags.DEFINE_string('cfm_log_path',
                    'D:/Source/PythonRepository/Projects/DeepLearningTechniques/ShakeNet/cifar10/cfm_log',
                    'Confusion Matrix 정보가 저장 되는 경로')

#todo 학습 시 사용되는 파라미터
flags.DEFINE_integer('epoch',
                     1800,
                     '훈련 시 총 에폭 수')

flags.DEFINE_integer('batch_size',
                     128,
                     '훈련 시 배치 크기')

flags.DEFINE_integer('step_per_epoch',
                     390,
                     '훈련 시 에폭 당 스텝 수')

flags.DEFINE_integer('image_width',
                     32,
                     '훈련 대상 이미지 가로 길이')

flags.DEFINE_integer('image_height',
                     32,
                     '훈련 대상 이미지 세로 길이')

flags.DEFINE_integer('image_channel',
                     3,
                     '훈련 대상 이미지 채널 수')

flags.DEFINE_float('dropout_rate',
                   0.4,
                   'dropout rate')

flags.DEFINE_float('drop_path_keep_prob',
                   0.6,
                   'drop-path rate')

flags.DEFINE_float('learning_rate',
                   0.2,
                   '신경망 learning rate')

flags.DEFINE_float('learning_rate_decay',
                   0.90,
                   '신경망 learning rate decay')

flags.DEFINE_float('l2_scale',
                   1e-4,
                   '신경망 L2 Regularization 크기')

flags.DEFINE_integer('image_class',
                     10,
                     '학습 대상 이미지의 클래스 개수')

flags.DEFINE_integer('decay_patience',
                     5,
                     'Learning rate decay 를 수행하는 조건')

flags.DEFINE_integer('earlystop_patience',
                     1800,
                     'Earlystop 을 수행하는 조건')