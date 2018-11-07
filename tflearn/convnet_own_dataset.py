# -*- coding: utf-8 -*-

""" Convolutional network applied to CIFAR-10 dataset classification task.

References:
    Learning Multiple Layers of Features from Tiny Images, A. Krizhevsky, 2009.

Links:
    [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

"""
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

# Data loading and preprocessing
from tflearn.data_utils import image_preloader

#from tflearn.datasets import cifar10
train_dataset_path = '/home/andrei/Data/Datasets/Scales/splited/train' 
val_dataset_path = '/home/andrei/Data/Datasets/Scales/splited/valid'
print('train_dataset_path:', train_dataset_path)
print('val_dataset_path:', val_dataset_path)

#(X, Y), (X_test, Y_test) = cifar10.load_data()
X, Y = image_preloader(train_dataset_path, mode='folder', image_shape=(32, 32),       #use the 'image_preloader'                    
	categorical_labels=True, normalize=True)
X_val, Y_val = image_preloader(val_dataset_path, mode='folder', image_shape=(32, 32),
	categorical_labels=True, normalize=True)
X_test, Y_test = X_val, Y_val
print('image_preloader done')

nb_classes = 148
X, Y = shuffle(X, Y)
Y = to_categorical(Y, nb_classes)
Y_test = to_categorical(Y_test, nb_classes)

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

print('Convolutional network building')
# Convolutional network building
network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=12, run_id='cifar10_cnn')
