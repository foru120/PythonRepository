from Projects.Hongbog.EyeVerification.keras.model import NeuralNet
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import load_model
import keras.backend as K
import keras
import tensorflow as tf
import math

image_scale = [(60, 160), (80, 200), (100, 240)]
right_gen_name = ['low_right_gen', 'mid_right_gen', 'high_right_gen']
left_gen_name = ['low_left_gen', 'mid_left_gen', 'high_left_gen']

train_right_gen, train_left_gen = dict(), dict()
test_right_gen, test_left_gen = dict(), dict()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=[0.9, 1.5],
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    fill_mode='nearest'
)

#todo Right Generator 설정
for gen_info in zip(right_gen_name, image_scale):
    train_right_gen[gen_info[0]] = train_datagen.flow_from_directory(
        directory='G:/04_dataset/eye_verification/eye_only_v3/train/right',
        target_size=gen_info[1],
        color_mode='grayscale',
        batch_size=50,
        class_mode='categorical',
        shuffle=True,
        seed=5
    )

    test_right_gen[gen_info[0]] = test_datagen.flow_from_directory(
        directory='G:/04_dataset/eye_verification/eye_only_v3/test/right',
        target_size=gen_info[1],
        color_mode='grayscale',
        batch_size=50,
        class_mode='categorical',
        shuffle=True,
        seed=5
    )

# todo Left Generator 설정
for gen_info in zip(right_gen_name, image_scale):
    train_left_gen[gen_info[0]] = train_datagen.flow_from_directory(
        directory='G:/04_dataset/eye_verification/eye_only_v3/train/left',
        target_size=gen_info[1],
        color_mode='grayscale',
        batch_size=50,
        class_mode='categorical',
        shuffle=True,
        seed=5
    )

    test_left_gen[gen_info[0]] = test_datagen.flow_from_directory(
        directory='G:/04_dataset/eye_verification/eye_only_v3/test/left',
        target_size=gen_info[1],
        color_mode='grayscale',
        batch_size=50,
        class_mode='categorical',
        shuffle=True,
        seed=5
    )

#todo multiple generator 설정
def train_multiple_gen():
    while True:
        low_x = train_right_gen['low_right_gen'].next()
        mid_x = train_right_gen['mid_right_gen'].next()
        high_x = train_right_gen['high_right_gen'].next()
        yield [low_x[0], mid_x[0], high_x[0]], low_x[1]

def test_multiple_gen():
    while True:
        low_x = test_right_gen['low_right_gen'].next()
        mid_x = test_right_gen['mid_right_gen'].next()
        high_x = test_right_gen['high_right_gen'].next()
        yield [low_x[0], mid_x[0], high_x[0]], low_x[1]

train_multiple_generator = train_multiple_gen()
test_multiple_generator = test_multiple_gen()

#todo custom callback 함수 설정
class CustomHistory(keras.callbacks.Callback):
    def init(self):
        super(CustomHistory, self).__init__()
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.train_acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))

#todo keras session 설정
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.5)
)

sess = tf.Session(config=config)
keras.backend.set_session(sess)

#todo model 생성
neuralnet = NeuralNet('right')
model = Model(inputs=[neuralnet.low_res_X, neuralnet.mid_res_X, neuralnet.high_res_X],
              outputs=neuralnet.prob)

#todo keras train 관련 설정
optimizer = optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#todo callback 함수 설정
'''
    checkpoint, tensorboard, earlystopping 기능 추가
'''
custom_hist = CustomHistory()
custom_hist.init()
callbacks = [
    custom_hist,
    keras.callbacks.ModelCheckpoint(
        filepath='D:/Source/PythonRepository/Projects/Hongbog/EyeVerification/keras/train_log/eye_verification.h5',
        monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min'
    ),
    keras.callbacks.TensorBoard(
        log_dir='D:/Source/PythonRepository/Projects/Hongbog/EyeVerification/keras/tensorboard',
        histogram_freq=0, write_graph=True, write_images=True
    ),
    keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=20, verbose=0, mode='auto')
]

#todo keras training
steps_per_epoch = math.ceil(train_right_gen['low_right_gen'].n / train_right_gen['low_right_gen'].batch_size)
validation_steps = math.ceil(test_right_gen['low_right_gen'].n / test_right_gen['low_right_gen'].batch_size)

print('-- Train --')
K.set_learning_phase(True)
model.fit_generator(
    generator=train_multiple_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    verbose=1,
    validation_data=test_multiple_generator,
    validation_steps=validation_steps,
    callbacks=callbacks
)

#todo keras evaluate
# tf.reset_default_graph()
#
# neuralnet = NeuralNet('right')
# model = Model(inputs=[neuralnet.low_res_X, neuralnet.mid_res_X, neuralnet.high_res_X],
#               outputs=neuralnet.prob)
#
# optimizer = optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0)
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#
# model.load_weights('D:/Source/PythonRepository/Projects/Hongbog/EyeVerification/keras/train_log/eye_verification.h5')

print('-- Evaluate --')
K.set_learning_phase(False)
eval_model = model.evaluate_generator(
    generator=test_multiple_generator,
    steps=validation_steps
)
print('Evaluation: ', eval_model)