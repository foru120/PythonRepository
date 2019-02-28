import tensorflow as tf

flags = tf.app.flags

#todo 학습 데이터/로그 경로
flags.DEFINE_string('dataset_name',
                    'cifar10',
                    '데이터 셋 이름')

flags.DEFINE_string('root_data_path',
                    'G:/04.dataset/02.cifar10/cifar10_tfrecord/cifar10',
                    '학습 데이터 경로 (cifar-10)')

flags.DEFINE_string('train_log_path',
                    'D:/Source/PythonRepository/Projects/Hongbog/MultiGPU/train_log/02',
                    '훈련 시 체크포인트 파일 저장 경로')

flags.DEFINE_string('deploy_log_path',
                    'D:/Source/PythonRepository/Projects/Hongbog/MultiGPU/deploy_log/02',
                    'Model Deploy 시 사용될 체크포인트 파일 저장 경로')

flags.DEFINE_string('cam_log_path',
                    'D:/Source/PythonRepository/Projects/Hongbog/MultiGPU/cam_log',
                    'CAM 이미지 및 체크포인트 파일 저장 경로')

flags.DEFINE_string('tensorboard_log_path',
                    'D:/Source/PythonRepository/Projects/Hongbog/MultiGPU/tensorboard_log/02',
                    'Tensor Board 에서 시각화하기 위해 저장되는 로그 경로')

flags.DEFINE_string('roc_curve_log_path',
                    'D:/Source/PythonRepository/Projects/Hongbog/MultiGPU/roc_curve_log',
                    'ROC Curve 그래프가 저장 되는 경로')

flags.DEFINE_string('cfm_log_path',
                    'D:/Source/PythonRepository/Projects/Hongbog/MultiGPU/cfm_log',
                    'Confusion Matrix 정보가 저장 되는 경로')

#todo 학습 시 사용되는 파라미터
flags.DEFINE_string('tower_name',
                    'tower',
                    'Multi-GPU 사용시 tower 이름')

flags.DEFINE_integer('epochs',
                     1800,
                     '훈련 시 총 에폭 수')

flags.DEFINE_integer('batch_size',
                     128,
                     '훈련 시 배치 크기')

flags.DEFINE_integer('num_gpus',
                     1,
                     '훈련 시 사용할 GPU 수')

flags.DEFINE_integer('image_width',
                     32,
                     '훈련 대상 이미지 가로 길이')

flags.DEFINE_integer('image_height',
                     32,
                     '훈련 대상 이미지 세로 길이')

flags.DEFINE_integer('image_channel',
                     3,
                     '훈련 대상 이미지 채널 수')

flags.DEFINE_float('lr',
                   0.2,
                   '신경망 learning rate')

flags.DEFINE_float('dropout_rate',
                   0.4,
                   '신경망 dropout_rate')

flags.DEFINE_float('l2_scale',
                   1e-4,
                   '신경망 L2 Regularization 크기')

flags.DEFINE_integer('num_classes',
                     10,
                     '학습 대상 이미지의 클래스 개수')