import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

#import tensorflow as tf

import math
import settings
train_dir = settings.data_dir + '/train'
valid_dir = settings.data_dir + '/valid'
train_batch_size = settings.train_batch_size
valid_batch_size = settings.valid_batch_size

IMAGE_SIZE = (224, 224)
INPUT_SHAPE = IMAGE_SIZE + (3,)

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
	target_size=IMAGE_SIZE,
	batch_size=train_batch_size,
	class_mode='categorical')

valid_generator = valid_datagen.flow_from_directory(
	valid_dir,
	target_size=IMAGE_SIZE,
	batch_size=train_batch_size,
	class_mode='categorical')

#--------------------------------------------
# model building

from keras import models
from keras import layers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.applications import VGG16, inception_v3
from keras.applications.inception_v3 import InceptionV3


input_tensor = keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
conv_base = InceptionV3(weights='imagenet', include_top=False, 
	input_tensor=input_tensor)

"""
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
#model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(settings.num_classes, activation='softmax'))
print(model.summary())
"""

x = conv_base.output
x = Flatten()(x)
#x = Dense(1024, activation="relu")(x)
#x = Dropout(0.5)(x)
#x = Dense(1024, activation="relu")(x)
#predictions = Dense(settings.num_classes, activation="softmax")(x)

predictions = layers.Dense(settings.num_classes, activation='softmax')(x)
model = Model(input=conv_base.input, output=predictions)

print('model.trainable_weights:', len(model.trainable_weights))
#conv_base.trainable = False

num_layers = len(model.layers)
num_last_trainable_layers = 60
for layer in model.layers[:num_layers-num_last_trainable_layers]:
    layer.trainable = False

print('model.trainable_weights:', len(model.trainable_weights))


"""
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=224*224*3))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(settings.num_classes, activation='softmax'))
"""

# ----------------------------------------------
# Training

#def top_6(y_true, y_pred):    
#    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=6)

model.compile(loss='categorical_crossentropy', #loss='binary_crossentropy',
			optimizer='adagrad', #optimizer=optimizers.RMSprop(lr=2e-5),
			metrics=['accuracy'])

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




