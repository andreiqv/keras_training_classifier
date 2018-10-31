import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from dataset_factory import GoodsDataset
import numpy as np

# import keras as K
OUTPUT_FOLDER = "output"
OUTPUT_MODEL_NAME = "se_recognizer"
#IMAGE_SIZE = (299, 299)
import settings
from settings import IMAGE_SIZE


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    # https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


callbacks = [
    # Interrupt training if `val_loss` stops improving for over 2 epochs
    # keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    # Write TensorBoard logs to `./logs` directory
    keras.callbacks.ModelCheckpoint(
        "./checkpoints/se_recognizer131018.{epoch:02d}-{val_loss:.2f}.hdf5"
    ),
    keras.callbacks.TensorBoard(
        log_dir='./tensorboard',
        write_images=True,
    )
]

#goods_dataset = GoodsDataset("dataset.list", OUTPUT_FOLDER + "/" + OUTPUT_MODEL_NAME + ".txt", 
#    IMAGE_SIZE, 20, 20, 5, 0.1)
goods_dataset = GoodsDataset("dataset-181018-ram.list", "dataset-181018.labels", (IMAGE_SIZE[0], IMAGE_SIZE[1]),
                           settings.train_batch, settings.valid_batch, 
                           settings.multiply, settings.valid_percentage)

input_tensor = keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
# input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
model = InceptionResNetV2(include_top=True, weights=None,
                          classes=goods_dataset.classes_count, input_tensor=input_tensor)

# tf.train.RMSPropOptimizer(0.001),
optimizer = keras.optimizers.RMSprop(lr=0.005)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())
# K.utils.print_summary(model, line_length=None, positions=None, print_fn=None)

# strategy = tf.contrib.distribute.MirroredStrategy()
# config = tf.estimator.RunConfig(train_distribute=strategy)

# keras_estimator = keras.estimator.model_to_estimator(
#   keras_model=model,
#   config=config,
#   model_dir='/tmp/model_dir')

# keras_estimator.train(input_fn=goods_dataset.get_train_dataset, steps=10)


model.fit(goods_dataset.get_train_dataset(),
          callbacks=callbacks,
          epochs=60,
          steps_per_epoch=800,  # 20 * 21 * 155 = 65100 < 13174 * 5 = 65870
          validation_data=goods_dataset.get_valid_dataset().repeat(),
          validation_steps=80)

session = keras.backend.get_session()
gr = session.graph.as_graph_def()
# for n in gr.node:
#     print(n.name)

# model.save_weights('./checkpoints/{}'.format(OUTPUT_MODEL_NAME))

# for n in keras.backend.get_session().graph.as_graph_def().node:
#     print(n.name)

frozen_graph = freeze_session(keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])

tf.train.write_graph(frozen_graph, OUTPUT_FOLDER, OUTPUT_MODEL_NAME + ".pb", as_text=False)
