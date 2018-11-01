# dataset parameters
IMAGE_SIZE = (299, 299)
multiply = 2
valid_percentage = 0.1
#train_batch = 32
#valid_batch = 32
train_batch = 10
valid_batch = 10
# 'sort' or 'shuffle'
dataset_order = 'shuffle' 
# a1 and b0 coeff. (max values), see https://www.tensorflow.org/api_docs/python/tf/contrib/image/transform
transform_maxval = 0.2 
rotation_max_angle = 90