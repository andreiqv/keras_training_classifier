# https://stackoverflow.com/questions/51622411/cant-import-frozen-graph-after-adding-layers-to-keras-model/51644241
# https://stackoverflow.com/questions/51858203/cant-import-frozen-graph-with-batchnorm-layer
# https://github.com/keras-team/keras/issues/11032
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import graph_io


def freeze_graph(graph, session, output):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, "./output", "inception_top60_181018-04-0.546-0.325[0.680]_rnd_adam.pb", as_text=False)


def top_6(y_true, y_pred):
    k = 6
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k)


keras.backend.set_learning_phase(0)

base_model = keras.models.load_model(
    "./output/top60_181018-04-0.546-0.325[0.680]_rnd_adam.hdf5",
    custom_objects={'top_6': top_6}
)

session = keras.backend.get_session()

print("model inputs")
for node in base_model.inputs:
    print(node.op.name)

print("model outputs")
for node in base_model.outputs:
    print(node.op.name)

freeze_graph(session.graph, session, [out.op.name for out in base_model.outputs])
