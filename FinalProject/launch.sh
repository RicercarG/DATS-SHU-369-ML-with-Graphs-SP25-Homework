#! /bin/bash

#SBATCH --job-name=train_gcn
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=48:00:00         # max time for running this job
#SBATCH --partition=sfscai
#SBATCH --gres=gpu:4          # request for any gpu available
#SBATCH --nodelist=gpu188
#SBATCH --output=logs/train_gcn-%j.out   # specify output directory

# remove SLURM_NTASKS
unset SLURM_NTASKS
export SLURM_NTASKS_PER_NODE=1

module load cuda/12.1.0-gcc-12.1.0 miniconda3/22.11.1
source activate /gpfsnyu/scratch/yg2709/env_g2p12

python train.py --no_instance --fineSize 256 --loadSize 256 --label_nc 0 --resize_or_crop resize_and_crop --output_nc 3 --batchSize 16 --mv --smart_disc --num_disc_images 7 --lr=0.0002