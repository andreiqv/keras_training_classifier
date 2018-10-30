import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import graph_io
from dataset_factory import GoodsDataset

# tf.enable_eager_execution()

def top_6(y_true, y_pred):
    k = 6
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k)


goods_dataset = GoodsDataset("dataset-181018.list", "dataset-181018.labels", (299, 299), 32, 32, 5, 0.1)


base_model = keras.models.load_model(
    "./output/inpcetionv3_top60_181018-02-0.876-0.584[0.881]_rnd_adagrad.hdf5",
    custom_objects={'top_6': top_6})

base_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy', top_6])

results = base_model.evaluate(goods_dataset.get_images_for_label(94).batch(16).repeat(), steps=6)
print(results)


# for i, (img, lbl) in enumerate(goods_dataset.get_ambroziya().batch(32).repeat()):
#     l = tf.argmax(lbl)
#     r = tf.math.equal(l, tf.constant(45, dtype=tf.int64))
#     print(i)