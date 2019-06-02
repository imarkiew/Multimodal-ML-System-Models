#!/usr/bin/env bash

python3 ./trainer/model.py \
    --data_dir ../data/train_and_val_data/processed_train_and_val_data \
    --output_dir ../results \
    --keras_save_period 2 \
    --train_batch_size 16 \
    --epochs 20 \
    --steps_per_epoch 402 \
    --val_batch_size 79 \
    --validation_steps 126 \
    --dropout_rate 0.5 \
    --learning_rate 0.00001 \
    --plateau_factor 0.5 \
    --plateau_patient 2 \
    --plateau_min_learning_rate 0.000001 \
    --number_of_classes 7 \
    --width 300 \
    --height 225 \
    --depth 3