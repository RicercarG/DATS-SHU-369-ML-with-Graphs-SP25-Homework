#! /bin/bash

module load cuda/12.1.0-gcc-12.1.0 miniconda3/22.11.1
source activate /gpfsnyu/scratch/yg2709/env_gnn


python Q3.py \
    --model_type 'fcnn' \
    --batch_size 128 \
    --epochs 80 \
    --learning_rate 1e-4 \