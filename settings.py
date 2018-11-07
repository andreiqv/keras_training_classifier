# dataset parameters
IMAGE_SIZE = (299, 299)
multiply = 5
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

dataset_list = 'dataset-181018.list'
labels_list = 'dataset-181018.labels'

#num_classes = 148
f = open(labels_list)
num_classes = len(['1' for x in f.readlines() if len(x.strip()) > 0])
print('num_classes:', num_classes)
