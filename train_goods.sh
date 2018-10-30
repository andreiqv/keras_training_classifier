#!/usr/bin/env bash
python3 retrain.py --image_dir /home/andrei/Data/Datasets/Scales/classifier_dataset_181018/424 \
    --output_graph /home/andrei/work/_output/se_classifier_121018.pb \
    --output_labels /home/andrei/work/_output/se_classifier_121018.txt \
    --summaries_dir /home/andrei/work/_output/tensorboard/ \
    --how_many_training_steps 2000000 \
    --learning_rate 0.001 \
    --tfhub_module https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1 \
    --saved_model_dir /home/andrei/work/_output/graph

#    --testing_percentage 1 \
#    --validation_percentage 5 \
