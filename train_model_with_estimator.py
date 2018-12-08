"""
https://github.com/mnoukhov/tf-estimator-mnist

"""

import tensorflow as tf
#import mnist_dataset as dataset
from utils.timer import timer

import settings
IMAGE_SIZE = (299, 299)
BATCH_SIZE = 32

#----------
flags = tf.app.flags
flags.DEFINE_string('data_dir', '/tmp/mnist/data',
					'Directory where mnist data will be downloaded'
					' if the data is not already there')
flags.DEFINE_string('model_dir', '/tmp/mnist/model',
					'Directory where all models are saved')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('num_epochs', 1,
					 'Num of batches to train (epochs).')
flags.DEFINE_float('learning_rate', 0.001, 'Learning Rate')
FLAGS = flags.FLAGS

#----------

from dataset_factory import GoodsDataset
goods_dataset = GoodsDataset("dataset-181018.list", "dataset-181018.labels", 
  settings.IMAGE_SIZE, settings.train_batch, settings.valid_batch, settings.multiply, 
  settings.valid_percentage)
train_dataset = goods_dataset.get_train_dataset()
valid_dataset = goods_dataset.get_valid_dataset()

def train_data():
	#data = dataset.train(FLAGS.data_dir)
	#data = data.cache()
	#data = data.batch(FLAGS.batch_size)
	#return data
	return train_dataset.prefetch(2).repeat()

def eval_data():
	#data = dataset.test(FLAGS.data_dir)
	#data = data.cache()
	#data = data.batch(FLAGS.batch_size)
	#return data
	return valid_dataset.repeat()

#-----------
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model

input_tensor = keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
conv_base = InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)
x = conv_base.output
x = Flatten()(x)
predictions = layers.Dense(settings.num_classes, activation='softmax')(x)
keras_model = lambda : Model(inputs=conv_base.input, outputs=predictions)


def model_function(features, labels, mode):
	# get the model
	model = keras_model

	if mode == tf.estimator.ModeKeys.TRAIN:
		# pass the input through the model
		logits = model(features)

		# get the cross-entropy loss and name it
		loss = tf.losses.sparse_softmax_cross_entropy(
			labels=labels,
			logits=logits)
		tf.identity(loss, 'train_loss')

		# record the accuracy and name it
		accuracy = tf.metrics.accuracy(
			labels=labels,
			predictions=tf.argmax(logits, axis=1))
		tf.identity(accuracy[1], name='train_accuracy')

		# use Adam to optimize
		optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
		tf.identity(FLAGS.learning_rate, name='learning_rate')

		# create an estimator spec to optimize the loss
		estimator_spec = tf.estimator.EstimatorSpec(
			mode=tf.estimator.ModeKeys.TRAIN,
			loss=loss,
			train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))

	elif mode == tf.estimator.ModeKeys.EVAL:
		# pass the input through the model
		logits = model(features, training=False)

		# get the cross-entropy loss
		loss = tf.losses.sparse_softmax_cross_entropy(
			labels=labels,
			logits=logits)

		# use the accuracy as a metric
		accuracy = tf.metrics.accuracy(
			labels=labels,
			predictions=tf.argmax(logits, axis=1))

		# create an estimator spec with the loss and accuracy
		estimator_spec = tf.estimator.EstimatorSpec(
			mode=tf.estimator.ModeKeys.EVAL,
			loss=loss,
			eval_metric_ops={
				'accuracy': accuracy
			})

	return estimator_spec


def main(_):
	hooks = [
		tf.train.LoggingTensorHook(
			['train_accuracy', 'train_loss'],
			every_n_iter=10)
	]

	NUM_GPUS = 3
	strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)
	config = tf.estimator.RunConfig(train_distribute=strategy)
	#estimator = tf.keras.estimator.model_to_estimator(model, config=config)

	mnist_classifier = tf.estimator.Estimator(
		model_fn=model_function,
		model_dir=FLAGS.model_dir,
		config=config
		)

	timer('TRAIN_AND_EVALUATE')

	for _ in range(FLAGS.num_epochs):
		mnist_classifier.train(
			input_fn=train_data,
			hooks=hooks,
		)
		mnist_classifier.evaluate(
			input_fn=eval_data)

	timer()

if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run()
