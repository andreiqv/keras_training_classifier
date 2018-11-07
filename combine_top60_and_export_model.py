# https://stackoverflow.com/questions/51622411/cant-import-frozen-graph-after-adding-layers-to-keras-model/51644241
# https://stackoverflow.com/questions/51858203/cant-import-frozen-graph-with-batchnorm-layer
# https://github.com/keras-team/keras/issues/11032
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import graph_io
import numpy as np
from dataset_factory import GoodsDataset

import settings
from settings import IMAGE_SIZE

model_name = 'top60_181018-07-0.996-0.664[0.924]_rnd_adam'


def top_6(y_true, y_pred):
    k = 6
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k)

def freeze_graph(graph, session, output):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen,\
            "./output", "inception_{}.pb".format(model_name), as_text=False)



keras.backend.set_learning_phase(0)

base_model = keras.models.load_model(
    "./output/{0}.hdf5".format(model_name),
    custom_objects={'top_6': top_6})

input_tensor = keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

main_model = keras.applications.InceptionV3(include_top=False,
                                            weights='imagenet',
                                            classes=settings.num_classes,
                                            pooling='avg',
                                            input_tensor=input_tensor)

predictions = keras.layers.Dense(settings.num_classes, activation='softmax')(main_model.output)
new_model = keras.Model(inputs=main_model.input, outputs=predictions)

new_model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', top_6])

session = keras.backend.get_session()

print(new_model.summary())

for i in range(0, len(base_model.layers)):
    new_model.layers[i + 248].set_weights(base_model.layers[i].get_weights())

"""
print("new model inputs")
for node in new_model.inputs:
    print(node.op.name)

print("new model outputs")
for node in new_model.outputs:
    print(node.op.name)
"""

do_evaluate = False
if do_evaluate:
    dataset = GoodsDataset("dataset-181018.list", "dataset-181018.labels", (IMAGE_SIZE[0], IMAGE_SIZE[1]),
                       32,
                       32, 5, 0.1)
    results = new_model.evaluate(dataset.get_valid_dataset(), steps=77)
    print(results)

new_model.save("output/inception_{0}.hdf5".format(model_name))

session = keras.backend.get_session()

print("model inputs")
for node in new_model.inputs:
    print(node.op.name)

print("model outputs")
for node in new_model.outputs:
    print(node.op.name)

print('num_layers:', len(new_model.layers))

freeze_graph(session.graph, session,\
    [out.op.name for out in new_model.outputs])
