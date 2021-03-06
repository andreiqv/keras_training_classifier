#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Прямой проход без хеширования промежуточных данных.
Последние 60 слоев берутся из inceptionv3_partial.py
Замораживаются первые несколько слоев сети.
"""

# https://github.com/tensorflow/tensorflow/issues/22837#issuecomment-428327601

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.densenet import DenseNet121

from tensorflow.keras.optimizers import SGD, Adam, Adagrad
from dataset_factory import GoodsDataset
import numpy as np
#from goods_tf_records import GoodsTfrecordsDataset

import nn_utils
from nn_utils import copy_model_weights

# tf.enable_eager_execution()
import settings
#from settings import IMAGE_SIZE
IMAGE_SIZE = (299, 299)
FREEZE_LAYERS = 618


def top_6(y_true, y_pred):
    k = 6
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k)

input_tensor = keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
#conv_base = InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)
#conv_base = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)
conv_base = VGG19(weights='imagenet', include_top=False, input_tensor=input_tensor)
#conv_base = DenseNet121(weights='imagenet', include_top=False, input_tensor=input_tensor)


"""
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
#model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(settings.num_classes, activation='softmax'))
print(model.summary())
"""

print('num_classes:', settings.num_classes)

x = conv_base.output
x = Flatten()(x)
predictions = layers.Dense(settings.num_classes, activation='softmax')(x)
model = Model(inputs=conv_base.input, outputs=predictions)


num_last_trainable_layers = None
num_layers = len(model.layers)
print('num_layers:', num_layers)
if num_last_trainable_layers:
  if num_last_trainable_layers >= 0 and num_last_trainable_layers < num_layers:
    for layer in model.layers[:num_layers-num_last_trainable_layers]:
      layer.trainable = False
print('num_last_trainable_layers:', num_last_trainable_layers)

print('model.trainable_weights:', len(model.trainable_weights))

#for layer in model.layers[249:]:
#  layer.trainable = True


# optimizer = keras.optimizers.RMSprop()
# optimizer = tf.train.GradientDescentOptimizer(0.2)

"""
from tensorflow.keras.utils import multi_gpu_model
model = multi_gpu_model(model, gpus=2)
"""

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.01), #Adam(lr=0.01), Adagrad(lr=0.01), #'adagrad',    #'rmsprop',   
              metrics=['accuracy', top_6])

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "./checkpoints/WHOLE_MODEL-{epoch:02d}-{acc:.3f}-{val_acc:.3f}[{val_top_6:.3f}].hdf5",
        save_best_only=True,
        monitor='val_top_6',
        mode='max'
    ),
    keras.callbacks.TensorBoard(
        log_dir='./tensorboard-whole',
        write_images=True,
    )
]

goods_dataset = GoodsDataset("dataset-181018.list", "dataset-181018.labels", 
  settings.IMAGE_SIZE, settings.train_batch, settings.valid_batch, settings.multiply, 
  settings.valid_percentage)
train_dataset = goods_dataset.get_train_dataset()
valid_dataset = goods_dataset.get_valid_dataset()

#results = model.evaluate(goods_dataset.get_images_for_label(94).batch(16).repeat(), steps=6)
#print(results)

model.fit(train_dataset.prefetch(2).repeat(), # was prefetch(2)
          callbacks=callbacks,
          epochs=200,
          steps_per_epoch=1157,
          validation_data=valid_dataset.repeat(),
          validation_steps=77,
          )


"""
1) VGG19: 
num_layers: 24
model.trainable_weights: 34

Epoch 1/200
868s 751ms/step - loss: 13.5427 - acc: 0.1592 - top_6: 0.9992 - val_loss: 13.4754 - val_acc: 0.1640 - val_top_6: 1.0000


"""