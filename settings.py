# dataset parameters
num_classes = 148
IMAGE_SIZE = (299, 299)
multiply = 1
valid_percentage = 0.1
train_batch = 32
valid_batch = 32
#train_batch = 32
#valid_batch = 32

# 'sort' or 'shuffle'
dataset_order = 'shuffle' 
# a1 and b0 coeff. (max values), see https://www.tensorflow.org/api_docs/python/tf/contrib/image/transform
transform_maxval = 0.2 
rotation_max_angle = 90