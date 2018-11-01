import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from dataset_factory import GoodsDataset
import numpy as np
from inceptionv3_partial import InceptionV3_top60, InceptionV3_top30
from goods_tf_records import InceptionV3Top60tfrecordsDataset

import settings
from settings import IMAGE_SIZE

#INPUT_SHAPE = (8, 8, 2048)   # top-30
#INPUT_SHAPE = (8, 8, 1280)   # top-60
INPUT_SHAPE = (299, 299, 3)   # whole network

def top_6(y_true, y_pred):
    k = 6
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k)

#dir_prefix = "/home/andrei/work/_tf_records"
#goods_dataset = InceptionV3Top60tfrecordsDataset(
#    dir_prefix + "/inceptionv3_top60_train_dataset1810.zlib.tfr",
#    dir_prefix + "/inceptionv3_top60_valid_dataset1810.zlib.tfr",
#    bottleneck_shape=INPUT_SHAPE
#)

class ImagesDataset: 
  
  def __init__(self):

    goods_dataset = GoodsDataset("dataset-181018.list", "dataset-181018.labels", 
        settings.IMAGE_SIZE, settings.train_batch, settings.valid_batch, settings.multiply, 
        settings.valid_percentage)
    train_set = goods_dataset.get_train_dataset()
    valid_set = goods_dataset.get_valid_dataset()    

    input_tensor = keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg',
                             input_tensor=input_tensor)
    output_layer_number=279  
    intermediate_layer_model = keras.Model(inputs=base_model.input,
                                           outputs=base_model.layers[output_layer_number].output)

    def _intermediate_processing(images, labels):
      images = intermediate_layer_model.predict(images, steps=77)
      return images, labels

    self.train_set = train_set.map(_intermediate_processing, num_parallel_calls=8)
    self.valid_set = valid_set.map(_intermediate_processing, num_parallel_calls=8)

goods_dataset = ImagesDataset()

inputs = keras.layers.Input(shape=INPUT_SHAPE)

# parent_model = keras.models.load_model(
#     "./output/inceptionv3-top30-0.83.hdf5",
#     custom_objects={'top_6': top_6})

model = InceptionV3_top60(inputs, 148)

print(model.summary())

for i, layer in enumerate(model.layers):
    print(i, layer.name)

# for i in range(1, len(model.layers)):
#     model.layers[i].set_weights(parent_model.layers[248 + i].get_weights())

model.compile(optimizer='adagrad',#'adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', top_6])

#print(model.evaluate(goods_dataset.valid_set.batch(32), steps=77))
print(model.evaluate(goods_dataset.valid_set, steps=77))

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

model.fit(goods_dataset.train_set.prefetch(10).repeat(),
          callbacks=callbacks,
          epochs=30,
          steps_per_epoch=1157,
          validation_data=goods_dataset.valid_set.repeat(),
          validation_steps=77,
          )
