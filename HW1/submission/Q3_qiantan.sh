#! /bin/bash

module load cuda/12.1.0-gcc-12.1.0 miniconda3/22.11.1
source activate /gpfsnyu/scratch/yg2709/env_gnn

## fcnn
python Q3.py \
    --model_type 'fcnn' \
    --batch_size 128 \
    --epochs 40 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4
# acc: 0.1052

python Q3.py \
    --model_type 'fcnn' \
    --batch_size 128 \
    --epochs 60 \
    --learning_rate 1e-4 \
    --weight_decay 1e-5
# acc: 0.0816

## cnn
python Q3.py \
    --model_type 'cnn' \
    --batch_size 128 \
    --epochs 40 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4
# acc: 0.1400

python Q3.py \
    --model_type 'cnn' \
    --batch_size 128 \
    --epochs 60 \
    --learning_rate 1e-4 \
    --weight_decay 1e-5
# acc: 0.1586

## lstm
python Q3.py \
    --model_type 'lstm' \
    --batch_size 128 \
    --epochs 40 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4
# acc: 0.0990

python Q3.py \
    --model_type 'lstm' \
    --batch_size 128 \
    --epochs 60 \
    --learning_rate 1e-4 \
    --weight_decay 1e-5
# acc: 0.0882