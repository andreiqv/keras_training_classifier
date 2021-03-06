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

"""
NotImplementedError: Only TF native optimizers are supported with DistributionStrategy.
"""
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

# for i in range(1, len(model.layers)):
#     model.layers[i].set_weights(parent_model.layers[248 + i].get_weights())

from tensorflow.python.training import gradient_descent
#optimizer = gradient_descent.GradientDescentOptimizer(0.01)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.2)
distribution = tf.contrib.distribute.MirroredStrategy()

model.compile(optimizer=optimizer, #Adagrad(lr=0.01), #'adagrad',#'adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', top_6],
              distribute=distribution,
              )

#print(model.evaluate(goods_dataset.valid_set.batch(32), steps=77))

"""
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
"""

model.fit(goods_dataset.train_set.batch(100).prefetch(10).repeat(),
          #callbacks=callbacks,
          epochs=20,
          steps_per_epoch=1157,
          validation_data=goods_dataset.valid_set.batch(32).repeat(),
          validation_steps=77,
          )

"""
1) без mirror strategy:
Epoch 1/20 - 199s 172ms/step - loss: 4.2596 - acc: 0.4212 - top_6: 0.7040 
                 - val_loss: 3.9561 - val_acc: 0.4460 - val_top_6: 0.7476

2) с mirror:
Epoch 1/20 - 356s 308ms/step - loss: 3.7931 - acc: 0.4881 - top_6: 0.7474 
- val_loss: 4.9878 - val_acc: 0.0225 - val_top_6: 0.0556



"""