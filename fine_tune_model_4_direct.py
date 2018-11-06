#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Прямой проход без хеширования промежуточных данных.
Последние 60 слоев берутся из inceptionv3_partial.py
Замораживаются первые 249 слоев сети InceptionV3.

"""

# https://github.com/tensorflow/tensorflow/issues/22837#issuecomment-428327601

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from dataset_factory import GoodsDataset
import numpy as np
from goods_tf_records import GoodsTfrecordsDataset

from nn_utils import copy_model_weights

# tf.enable_eager_execution()
import settings
#from settings import IMAGE_SIZE
IMAGE_SIZE = (299, 299)
FREEZE_LAYERS = 618


def top_6(y_true, y_pred):
    k = 6
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k)

# goods_dataset = GoodsDataset("dataset.list", "output/se_classifier_161018.list", IMAGE_SIZE, 32, 32, 5, 0.1)

#input_tensor = keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
#
#model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg', 
#  input_tensor=input_tensor)

#base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg', 
#  input_tensor=input_tensor)
#x = base_model.output
#predictions = layers.Dense(settings.num_classes, activation='softmax')(x)
#model = keras.Model(inputs=base_model.input, outputs=predictions)

from nn import get_InceptionV3_whole_model

model = get_InceptionV3_whole_model()

print(model.summary())

source_model = keras.models.load_model(
    "./output/inception_top60_181018-03-0.869-0.700[0.950]_rnd_adam.hdf5",
    custom_objects={'top_6': top_6}
)

start_training_layer=249
copy_model_weights(source_model, model, start_layer=start_training_layer)
for layer in model.layers[:start_training_layer]:
  layer.trainable = False


#for layer in model.layers[249:]:
#  layer.trainable = True


# for i, layer in enumerate(base_model.layers):
#     print(i, layer.name)

# path_prefix = "/home/chichivica/Data/Datasets/ramdisk/"
#goods_dataset = GoodsTfrecordsDataset("./tf_records/inceptionv3_bottlenecks_train_dataset181018.zlib.tfr",
#                                      "./tf_records/inceptionv3_bottlenecks_valid_dataset181018.zlib.tfr",
#                                      32, 32, 'ZLIB')

#model = keras.Sequential()
#model.add(keras.layers.Flatten(input_shape=(2048,)))
#model.add(keras.layers.Flatten(input_shape=(2048,)))
#model.add(keras.layers.Dense(148, activation='softmax'))


# optimizer = keras.optimizers.RMSprop()
# optimizer = tf.train.GradientDescentOptimizer(0.2)

model.compile(optimizer='adagrad',    #'rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy', top_6])

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "./checkpoints/inceptionv3-181018-{epoch:02d}-{acc:.3f}-{val_acc:.3f}[{val_top_6:.3f}].hdf5",
        save_best_only=True,
        monitor='val_top_6',
        mode='max'
    ),
    keras.callbacks.TensorBoard(
        log_dir='./tensorboard-incv4',
        write_images=True,
    )
]


goods_dataset = GoodsDataset("dataset-181018.list", "dataset-181018.labels", 
  settings.IMAGE_SIZE, settings.train_batch, settings.valid_batch, settings.multiply, 
  settings.valid_percentage)
train_dataset = goods_dataset.get_train_dataset()
valid_dataset = goods_dataset.get_valid_dataset()   

model.fit(train_dataset.prefetch(2).repeat(),
          callbacks=callbacks,
          epochs=300,
          steps_per_epoch=1157,
          validation_data=valid_dataset.repeat(),
          validation_steps=77,
          )
