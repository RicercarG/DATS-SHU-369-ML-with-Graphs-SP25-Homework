module load cuda/12.1.0-gcc-12.1.0 miniconda3/22.11.1
source activate /gpfsnyu/scratch/yg2709/env_g2p12

python gen_imgs.py --dataroot ./datasets/newbreeder/ --name art_newbreeder_allparents_bs8_disc --no_instance --fineSize 256 --loadSize 256 --label_nc 0 --resize_or_crop resize_and_crop --output_nc 3 --batchSize 8 --mv --smart_disc --num_disc_images 7 --which_epoch 50