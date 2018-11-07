# keras_training_classifier
keras training github

1) find $PWD -type f | grep -i .jpg > dataset-181018.list

mv dataset-181018.list .

2) python3 dataset_factory.py --data dataset-181018.list --labels dataset-181018.labels

3) python3 goods_tf_records.py 

4) python3 fine_tune_model.py
