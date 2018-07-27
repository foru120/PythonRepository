import numpy as np

# 랜덤시트 고정시키기
np.random.seed(3)

from keras.preprocessing.image import ImageDataGenerator

# 데이터셋 불러오기
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.7,
    zoom_range=[0.9, 2.2],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    directory='./dataset/handwriting_shape/train',
    target_size=(24, 24),
    batch_size=3,
    class_mode='categorical'
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

test_generator = test_datagen.flow_from_directory(
    directory='./dataset/handwriting_shape/test',
    target_size=(24, 24),
    batch_size=3,
    class_mode='categorical'
)

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.layers import Dropout

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 3)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 모델 엮기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습시키기
model.fit_generator(
    generator=train_generator,
    steps_per_epoch=15 * 100,
    epochs=200,
    validation_data=test_generator,
    validation_steps=5
)

# 모델 평가하기
print('-- Evaluate --')

scores = model.evaluate_generator(
    test_generator,
    steps=5
)

print('%s: %.2f%%' %(model.metrics_names[1], scores[1] * 100))

# 모델 예측하기
print('-- Predict --')

output = model.predict_generator(
    test_generator,
    steps=5
)

np.set_printoptions(formatter={'float': lambda x: '{0:0.3f}'.format(x)})

print(output)