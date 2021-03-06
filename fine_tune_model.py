import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from dataset_factory import GoodsDataset
import numpy as np
from inceptionv3_partial import InceptionV3_top60, InceptionV3_top30
from goods_tf_records import InceptionV3Top60tfrecordsDataset
from tensorflow.keras.optimizers import SGD, Adam, Adagrad

import settings


def top_6(y_true, y_pred):
    k = 6
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k)


IMAGE_SIZE = settings.IMAGE_SIZE
#INPUT_SHAPE = (8, 8, 2048)   # top-30
INPUT_SHAPE = (8, 8, 1280)   # top-60

dir_prefix = "/home/andrei/work/_tf_records"
goods_dataset = InceptionV3Top60tfrecordsDataset(
    dir_prefix + "/inceptionv3_top60_train_dataset1810.zlib.tfr",
    dir_prefix + "/inceptionv3_top60_valid_dataset1810.zlib.tfr",
    bottleneck_shape=INPUT_SHAPE
)

inputs = keras.layers.Input(shape=INPUT_SHAPE)

# parent_model = keras.models.load_model(
#     "./output/inceptionv3-top30-0.83.hdf5",
#     custom_objects={'top_6': top_6})

model = InceptionV3_top60(inputs, settings.num_classes)

print(model.summary())

for i, layer in enumerate(model.layers):
    print(i, layer.name)
print('model.trainable_weights:', len(model.trainable_weights))

# for i in range(1, len(model.layers)):
#     model.layers[i].set_weights(parent_model.layers[248 + i].get_weights())

model.compile(optimizer='adagrad', #'adagrad',#'adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', top_6])

print(model.evaluate(goods_dataset.valid_set.batch(32), steps=77))

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "./checkpoints/top60_181018-{epoch:02d}-{acc:.3f}-{val_acc:.3f}[{val_top_6:.3f}]_rnd_adam.hdf5",
        save_best_only=True,
        monitor='val_top_6',
        mode='max'
    ),
    keras.callbacks.TensorBoard(
        log_dir='./tensorboard-top60',
    )
]

model.fit(goods_dataset.train_set.batch(100).prefetch(5).repeat(),
          callbacks=callbacks,
          epochs=10,
          steps_per_epoch=1157,
          validation_data=goods_dataset.valid_set.batch(32).repeat(),
          validation_steps=77,
          )


#-----------------------------------------------

num_freeze_layers = 50
for layer in model.layers[:num_freeze_layers]:
    layer.trainable = False
print('the first {} layers was frozen'.format(num_freeze_layers))
print('model.trainable_weights:', len(model.trainable_weights))

model.compile(optimizer=Adagrad(lr=0.002), #'adagrad',#'adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', top_6])
model.fit(goods_dataset.train_set.batch(100).prefetch(5).repeat(),
          callbacks=callbacks,
          epochs=5,
          steps_per_epoch=1157,
          validation_data=goods_dataset.valid_set.batch(32).repeat(),
          validation_steps=77,
          )