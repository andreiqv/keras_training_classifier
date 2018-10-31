# dataset parameters
IMAGE_SIZE = (299, 299)
multiply = 5
valid_percentage = 0.1
#train_batch = 32
#valid_batch = 32
train_batch = 8
valid_batch = 8
dataset_order = 'shuffle' # 'sort' or 'shuffle'
transform_maxval = 0.2 # a1 and b0 coeff. (max values), see https://www.tensorflow.org/api_docs/python/tf/contrib/image/transform
rotation_max_angle = 90