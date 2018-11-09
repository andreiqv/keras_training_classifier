import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3

from tensorflow.keras.optimizers import SGD, Adam, Adagrad
from tensorflow.python.training import gradient_descent
from tensorflow.keras.utils import multi_gpu_model

import numpy as np
from dataset_factory import GoodsDataset
from inceptionv3_partial import InceptionV3_top60, InceptionV3_top30
from goods_tf_records import InceptionV3Top60tfrecordsDataset
import settings

#optimizer = gradient_descent.GradientDescentOptimizer(0.001)

"""
NotImplementedError: Only TF native optimizers are supported with DistributionStrategy.
"""



def top_6(y_true, y_pred):
    k = 6
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k)


IMAGE_SIZE = settings.IMAGE_SIZE
#INPUT_SHAPE = (8, 8, 2048)   # top-30

# parent_model = keras.models.load_model(
#     "./output/inceptionv3-top30-0.83.hdf5",
#     custom_objects={'top_6': top_6})

input_tensor = keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
#
#model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg', 
#  input_tensor=input_tensor)

base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg', 
  input_tensor=input_tensor)
x = base_model.output
#x = GlobalAveragePooling2D()(x)
#x = layers.Dense(148, activation='relu')(x)
predictions = layers.Dense(settings.num_classes, activation='softmax')(x)
model = keras.Model(inputs=base_model.input, outputs=predictions)
print(model.summary())
start_training_layer = 249

print(model.summary())

for i, layer in enumerate(model.layers):
    print(i, layer.name)

# for i in range(1, len(model.layers)):
#     model.layers[i].set_weights(parent_model.layers[248 + i].get_weights())

#distribution = tf.contrib.distribute.MirroredStrategy()

G = 2
model = multi_gpu_model(model, gpus=G)

model.compile(optimizer='adagrad', #Adagrad(lr=0.01), #'adagrad',#'adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', top_6])
              #distribute=distribution)

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

#print(model.evaluate(goods_dataset.valid_set.batch(32), steps=77))

model.fit(train_dataset.prefetch(2).repeat(),
          callbacks=callbacks,
          epochs=300,
          steps_per_epoch=1157,
          validation_data=valid_dataset.repeat(),
          validation_steps=77,
          )

