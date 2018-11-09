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
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
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
starting_copied_layer = 249

source_top60_model = keras.models.load_model(
    "./output/top60_181018-03-0.869-0.700[0.950]_rnd_adam.hdf5",
    custom_objects={'top_6': top_6}
)

#nn_utils.copy_model_weights(source_model, model, start_layer=start_training_layer)
nn_utils.copy_top_weights_to_model(source_top60_model, model, start_layer=starting_copied_layer)
print('Weights of top60 was copied.')

#for layer in model.layers[:init_copied_layer]:
#    layer.trainable = False

num_layers = len(model.layers)
num_last_trainable_layers = 30
for layer in model.layers[:num_layers-num_last_trainable_layers]:
    layer.trainable = False

print('num_last_trainable_layers:', num_last_trainable_layers)
print('model.trainable_weights:', len(model.trainable_weights))

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

from tensorflow.keras.utils import multi_gpu_model
model = multi_gpu_model(model, gpus=2)

model.compile(optimizer=Adagrad(lr=0.01), #'adagrad',    #'rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy', top_6])

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "./checkpoints/FINE_TUNE_MODEL_4_DIRECT_inceptionv3-181018-{epoch:02d}-{acc:.3f}-{val_acc:.3f}[{val_top_6:.3f}].hdf5",
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

results = model.evaluate(goods_dataset.get_images_for_label(94).batch(16).repeat(), steps=6)
print(results)

model.fit(train_dataset.prefetch(5).repeat(), # was prefetch(2)
          callbacks=callbacks,
          epochs=30,
          steps_per_epoch=1157,
          validation_data=valid_dataset.repeat(),
          validation_steps=77,
          )


"""
1) num_last_trainable_layers = 60 
optimizer=Adagrad(lr=0.001) 
(новая аугментация с вращ и транс. - 734s 634ms/step)
Epoch 1/50 - loss: 1.9875 - acc: 0.4700 - top_6: 0.8023 - val_loss: 2.9080 - val_acc: 0.4233 - val_top_6: 0.7297
Epoch 2/50 - loss: 1.7136 - acc: 0.5227 - top_6: 0.8451 - val_loss: 2.9081 - val_acc: 0.4110 - val_top_6: 0.7262
Epoch 3/50 - loss: 1.6264 - acc: 0.5377 - top_6: 0.8615 - val_loss: 2.9565 - val_acc: 0.4118 - val_top_6: 0.7185
Epoch 4/50 - loss: 1.5851 - acc: 0.5468 - top_6: 0.8658 - val_loss: 2.8856 - val_acc: 0.4281 - val_top_6: 0.7226
Epoch 5/50 - loss: 1.5577 - acc: 0.5500 - top_6: 0.8708 - val_loss: 2.9554 - val_acc: 0.4150 - val_top_6: 0.7165


(старая аугментация  344s 297ms/step)
Adagrad(lr=0.0002) 
Epoch 1/50 - loss: 1.1296 - acc: 0.6702 - top_6: 0.9373 - val_loss: 3.9620 - val_acc: 0.3709 - val_top_6: 0.7147
Epoch 2/50 - loss: 0.9970 - acc: 0.6970 - top_6: 0.9491 - val_loss: 3.9317 - val_acc: 0.3688 - val_top_6: 0.7107
Epoch 3/50 - loss: 0.9595 - acc: 0.7074 - top_6: 0.9526 - val_loss: 3.9356 - val_acc: 0.3660 - val_top_6: 0.7127
Epoch 4/50 - loss: 0.9318 - acc: 0.7146 - top_6: 0.9559 - val_loss: 3.9380 - val_acc: 0.3595 - val_top_6: 0.7099
Epoch 5/50 - loss: 0.9098 - acc: 0.7200 - top_6: 0.9582 - val_loss: 3.9248 - val_acc: 0.3599 - val_top_6: 0.7091

2) num_last_trainable_layers = 5
Cтарая аугментация:
optimizer=Adagrad(lr=0.001)
Epoch 1/30 - loss: 1.4343 - acc: 0.6053 - top_6: 0.9029 - val_loss: 1.0382 - val_acc: 0.6989 - val_top_6: 0.9485
Epoch 2/30 - loss: 1.3089 - acc: 0.6311 - top_6: 0.9137 - val_loss: 1.0326 - val_acc: 0.7004 - val_top_6: 0.9464
Epoch 3/30 - loss: 1.2674 - acc: 0.6352 - top_6: 0.9199 - val_loss: 1.0301 - val_acc: 0.6996 - val_top_6: 0.9440
Epoch 4/30 - loss: 1.2353 - acc: 0.6449 - top_6: 0.9231 - val_loss: 1.0360 - val_acc: 0.7004 - val_top_6: 0.9460

Новая аугментация:
optimizer=Adagrad(lr=0.001)
Epoch 1/30 - loss: 1.8551 - acc: 0.5239 - top_6: 0.8482 - val_loss: 1.0392 - val_acc: 0.7033 - val_top_6: 0.9489
Epoch 2/30 - loss: 1.7007 - acc: 0.5512 - top_6: 0.8660 - val_loss: 1.0471 - val_acc: 0.6934 - val_top_6: 0.9468
Epoch 3/30 - loss: 1.6342 - acc: 0.5606 - top_6: 0.8739 - val_loss: 1.0452 - val_acc: 0.6946 - val_top_6: 0.9452
Epoch 4/30 - loss: 1.5976 - acc: 0.5637 - top_6: 0.8795 - val_loss: 1.0364 - val_acc: 0.6954 - val_top_6: 0.9488
Epoch 5/30 - loss: 1.5738 - acc: 0.5695 - top_6: 0.8800 - val_loss: 1.0426 - val_acc: 0.6942 - val_top_6: 0.9460
Epoch 6/30 - loss: 1.5458 - acc: 0.5740 - top_6: 0.8831 - val_loss: 1.0427 - val_acc: 0.6921 - val_top_6: 0.9460
Epoch 7/30 - loss: 1.5330 - acc: 0.5778 - top_6: 0.8853 - val_loss: 1.0408 - val_acc: 0.6958 - val_top_6: 0.9452
Epoch 8/30 - loss: 1.5175 - acc: 0.5772 - top_6: 0.8862 - val_loss: 1.0512 - val_acc: 0.6913 - val_top_6: 0.9448

---------

3) Новая аугментация
num_last_trainable_layers = 60
optimizer=Adagrad(lr=0.001)

Epoch 1/30 - loss: 1.2307 - acc: 0.6386 - top_6: 0.9250 - val_loss: 3.5862 - val_acc: 0.3649 - val_top_6: 0.7196
Epoch 2/30 - loss: 1.0732 - acc: 0.6739 - top_6: 0.9410 - val_loss: 3.4367 - val_acc: 0.3759 - val_top_6: 0.7309
Epoch 10/30 - loss: 0.8951 - acc: 0.7235 - top_6: 0.9600 - val_loss: 3.0816 - val_acc: 0.4007 - val_top_6: 0.7614
Epoch 28/30 - loss: 0.7834 - acc: 0.7549 - top_6: 0.9703 - val_loss: 2.8955 - val_acc: 0.4104 - val_top_6: 0.7772


optimizer=Adagrad(lr=0.01)
Epoch 1/30 - loss: 1.4073 - acc: 0.6039 - top_6: 0.9043 - val_loss: 3.1335 - val_acc: 0.3705 - val_top_6: 0.7110
Epoch 2/30 - loss: 1.0566 - acc: 0.6791 - top_6: 0.9431 - val_loss: 2.9609 - val_acc: 0.3895 - val_top_6: 0.7412
Epoch 3/30 - loss: 0.9562 - acc: 0.7067 - top_6: 0.9535 - val_loss: 3.1973 - val_acc: 0.3521 - val_top_6: 0.7152

4) num_last_trainable_layers = 30
optimizer=Adagrad(lr=0.01)
model.trainable_weights: 18
1592s 1s/step 
Epoch 1/30 - loss: 1.4455 - acc: 0.5998 - top_6: 0.9004 - val_loss: 1.1997 - val_acc: 0.6485 - val_top_6: 0.9278
Epoch 2/30 - loss: 1.1181 - acc: 0.6633 - top_6: 0.9357 - val_loss: 1.1849 - val_acc: 0.6630 - val_top_6: 0.9334
Epoch 3/30 - loss: 1.0501 - acc: 0.6793 - top_6: 0.9448 - val_loss: 1.1757 - val_acc: 0.6662 - val_top_6: 0.9310
Epoch 6/30 - loss: 0.9210 - acc: 0.7164 - top_6: 0.9569 - val_loss: 1.1942 - val_acc: 0.6593 - val_top_6: 0.9294
Epoch 10/30- loss: 0.8422 - acc: 0.7362 - top_6: 0.9657 - val_loss: 1.1846 - val_acc: 0.6710 - val_top_6: 0.9290

"""