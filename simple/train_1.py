from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

import math
import settings
train_dir = settings.data_dir + '/train'
valid_dir = settings.data_dir + '/valid'
train_batch_size = settings.train_batch_size
valid_batch_size = settings.valid_batch_size

#------------------------
# data preparing

IMAGE_SHAPE = (150,150)
num_classes = 148

train_datagen = ImageDataGenerator(
	rescale=1./255,
	rotation_range=40,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,
	fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
	train_dir,
	target_size=IMAGE_SHAPE,
	batch_size=train_batch_size,
	class_mode='binary')

valid_generator = valid_datagen.flow_from_directory(
	valid_dir,
	target_size=IMAGE_SHAPE,
	batch_size=train_batch_size,
	class_mode='binary')

#--------------------------------------------
# model building

from keras import models
from keras import layers
from keras.applications import VGG16

"""
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
print(model.summary())
"""

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=(3,) + IMAGE_SHAPE))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(layers.Dense(64))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes))
#model.add(layers.Activation('sigmoid'))


print('model.trainable_weights:', len(model.trainable_weights))
conv_base.trainable = False
print('model.trainable_weights:', len(model.trainable_weights))

# ----------------------------------------------
# Training


model.compile(loss='categorical_crossentropy', # 'binary_crossentropy',
			optimizer='rmsprop', # optimizer=optimizers.RMSprop(lr=2e-5),
			metrics=['acc'])

train_steps_per_epoch = math.ceil(train_generator.n / train_generator.batch_size)
validation_steps = math.ceil(valid_generator.n / valid_generator.batch_size)
print('train data size:', train_generator.n)
print('train steps per epoch:', train_steps_per_epoch)
print('valid data size:', valid_generator.n)
print('validation_steps:', validation_steps)

history = model.fit_generator(
	train_generator,
	steps_per_epoch=train_steps_per_epoch,
	epochs=30,
	validation_data=valid_generator,
	validation_steps=validation_steps)




