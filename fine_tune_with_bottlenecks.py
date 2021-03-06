# https://github.com/tensorflow/tensorflow/issues/22837#issuecomment-428327601

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from dataset_factory import GoodsDataset
import numpy as np
from goods_tf_records import GoodsTfrecordsDataset


# tf.enable_eager_execution()


def top_6(y_true, y_pred):
    k = 6
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k)


IMAGE_SIZE = (299, 299)
FREEZE_LAYERS = 618

# goods_dataset = GoodsDataset("dataset.list", "output/se_classifier_161018.list", IMAGE_SIZE, 32, 32, 5, 0.1)

# input_tensor = keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
#
# base_model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg', input_tensor=input_tensor)

# print(base_model.summary())

# for i, layer in enumerate(base_model.layers):
#     print(i, layer.name)

# path_prefix = "/home/chichivica/Data/Datasets/ramdisk/"
goods_dataset = GoodsTfrecordsDataset("./tf_records/inceptionv3_bottlenecks_train_dataset181018.zlib.tfr",
                                      "./tf_records/inceptionv3_bottlenecks_valid_dataset181018.zlib.tfr",
                                      32, 32, 'ZLIB')
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(2048,)))
model.add(keras.layers.Dense(148, activation='softmax'))

# optimizer = keras.optimizers.RMSprop()
# optimizer = tf.train.GradientDescentOptimizer(0.2)

model.compile(optimizer='rmsprop',
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

model.fit(goods_dataset.train_set.batch(100).prefetch(2).repeat(),
          callbacks=callbacks,
          epochs=300,
          steps_per_epoch=1157,
          validation_data=goods_dataset.valid_set.batch(32).repeat(),
          validation_steps=77,
          )
