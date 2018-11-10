#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Прямой проход без хеширования промежуточных данных.
Последние 60 слоев берутся из inceptionv3_partial.py
Замораживаются первые 249 слоев сети InceptionV3.

Epoch 1/300
1157/1157 [==============================] - 1105s 955ms/step 
- loss: 1.5288 - acc: 0.5829 - top_6: 0.8920 
- val_loss: 3.2239 - val_acc: 0.3697 - val_top_6: 0.6985



--
all false trainable:
[1.639923135439555, 0.3680555572112401, 0.9791666666666666]
Epoch 1/50 - 1055s 912ms/step - loss: 2.1915 - acc: 0.4807 - top_6: 0.8107 
 - val_loss: 1.0686 - val_acc: 0.6972 - val_top_6: 0.9501
Epoch 2/50 - 992s 857ms/step - loss: 2.1883 - acc: 0.4818 - top_6: 0.8103 
 - val_loss: 1.0649 - val_acc: 0.6983 - val_top_6: 0.9505
Epoch 3/50 - 994s 859ms/step - loss: 2.2185 - acc: 0.4771 - top_6: 0.8094 
 - val_loss: 1.0639 - val_acc: 0.6995 - val_top_6: 0.9509


обучать последние 10 слоев:
Epoch 1/50 - loss: 1.6911 - acc: 0.5556 - top_6: 0.8710 - val_loss: 1.1406 - val_acc: 0.6757 - val_top_6: 0.9351
Epoch 2/50 - loss: 1.4247 - acc: 0.5921 - top_6: 0.8971 - val_loss: 1.1161 - val_acc: 0.6742 - val_top_6: 0.9357
Epoch 3/50 - loss: 1.3506 - acc: 0.6108 - top_6: 0.9057 - val_loss: 1.1025 - val_acc: 0.6754 - val_top_6: 0.9378
Epoch 4/50 - loss: 1.3249 - acc: 0.6150 - top_6: 0.9090 - val_loss: 1.0839 - val_acc: 0.6823 - val_top_6: 0.9390
Epoch 5/50 - loss: 1.3133 - acc: 0.6145 - top_6: 0.9110 - val_loss: 1.0971 - val_acc: 0.6823 - val_top_6: 0.9365

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
conv_base = InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)
#conv_base = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)

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


num_layers = len(model.layers)
print('num_layers:', num_layers)

#num_last_trainable_layers = 100
#for layer in model.layers[:num_layers-num_last_trainable_layers]:
#   layer.trainable = False
#print('num_last_trainable_layers:', num_last_trainable_layers)

print('model.trainable_weights:', len(model.trainable_weights))

#for layer in model.layers[249:]:
#  layer.trainable = True


# optimizer = keras.optimizers.RMSprop()
# optimizer = tf.train.GradientDescentOptimizer(0.2)

from tensorflow.keras.utils import multi_gpu_model
model = multi_gpu_model(model, gpus=2)

model.compile(optimizer=Adagrad(lr=0.01), #'adagrad',    #'rmsprop',
              loss='categorical_crossentropy',
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

results = model.evaluate(goods_dataset.get_images_for_label(94).batch(16).repeat(), steps=6)
print(results)

model.fit(train_dataset.prefetch(16).repeat(), # was prefetch(2)
          callbacks=callbacks,
          epochs=200,
          steps_per_epoch=1157,
          validation_data=valid_dataset.repeat(),
          validation_steps=77,
          )


"""


"""