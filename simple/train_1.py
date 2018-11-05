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
	target_size=(150,150),
	batch_size=train_batch_size,
	class_mode='binary')

valid_generator = valid_datagen.flow_from_directory(
	valid_dir,
	target_size=(150,150),
	batch_size=train_batch_size,
	class_mode='binary')

#--------------------------------------------
# model building

from keras import models
from keras import layers
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
print(model.summary())

print('model.trainable_weights:', len(model.trainable_weights))
conv_base.trainable = False
print('model.trainable_weights:', len(model.trainable_weights))

# ----------------------------------------------
# Training


model.compile(loss='binary_crossentropy',
			optimizer=optimizers.RMSprop(lr=2e-5),
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




