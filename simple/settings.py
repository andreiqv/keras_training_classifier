#export CUDA_VISIBLE_DEVICES=3
import os

if os.path.exists('.local'):
	data_dir = '/w/WORK/ineru/06_scales/_dataset/splited'
	train_batch_size = 2
	valid_batch_size = 2

else:
	data_dir = '/home/andrei/Data/Datasets/Scales/splited'
	train_batch_size = 32
	valid_batch_size = 32
