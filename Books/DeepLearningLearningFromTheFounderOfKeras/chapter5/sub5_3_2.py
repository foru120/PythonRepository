#todo p.211 ~ p.218
#todo code 5-22 ~ code 5-24
#todo 5.3.2 미세 조정

import os
import matplotlib.pyplot as plt

from keras.models import load_model
from keras import models, layers, optimizers
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator

base_dir = '/home/kyh/dataset/dogs_and_cats_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# conv_base = VGG16(include_top=False,
#                   weights='imagenet',
#                   input_shape=(150, 150, 3))
#
# model = models.Sequential()
# model.add(conv_base)
# model.add(layers.Flatten())
# model.add(layers.Dense(units=256, activation='relu'))
# model.add(layers.Dense(units=1, activation='sigmoid'))

tot_model = load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'train_dir',
                                    'cats_and_dogs_small_5_22.h5'))
vgg_model = tot_model.get_layer('vgg16')
vgg_model.trainable = True

tot_model.trainable = True

set_trainable = False
for layer in vgg_model.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    directory=validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

test_generator = validation_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

tot_model.compile(optimizer=optimizers.RMSprop(lr=1e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = tot_model.fit_generator(
    generator=train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50,
    verbose=2
)

tot_model.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'train_dir',
                            'cats_and_dogs_small_5_22_v2.h5'))

acc = history.history['acc']
loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:  # smoothed_points 리스트에 원소가 있을 경우
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

plt.plot(epochs,
         smooth_curve(acc), 'bo', label='Training acc')
plt.plot(epochs,
         smooth_curve(val_acc), 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs,
         smooth_curve(loss), 'bo', label='Training loss')
plt.plot(epochs,
         smooth_curve(val_loss), 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test_acc:', test_acc)