module load cuda/12.1.0-gcc-12.1.0 miniconda3/22.11.1
source activate /gpfsnyu/scratch/yg2709/env_g2p12

echo "Calculating FID score..."
echo "CFG 7.5"
python -m pytorch_fid gen_images/art_newbreeder_allparents_bs8_disc/gts gen_images/art_newbreeder_allparents_bs8_disc/generated/7.5
echo "=========================="
echo "CFG 3"
python -m pytorch_fid gen_images/art_newbreeder_allparents_bs8_disc/gts gen_images/art_newbreeder_allparents_bs8_disc/generated/3
echo "=========================="
echo "CFG 1"
python -m pytorch_fid gen_images/art_newbreeder_allparents_bs8_disc/gts gen_images/art_newbreeder_allparents_bs8_disc/generated/1