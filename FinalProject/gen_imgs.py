import os

import numpy as np
import torch
import torchvision
import lightning as L
from tqdm import tqdm

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import GC_Diffusion
import util.util as util

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 8  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip


data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

# # load the model weights from the checkpoint
checkpoint_path = "checkpoints/last.ckpt"
model = GC_Diffusion.load_from_checkpoint(checkpoint_path, opt=opt)

BASE_DIR = f"gen_images/{opt.name}/"
GT_DIR = f"{BASE_DIR}gts"
PRED_DIR = f"{BASE_DIR}generated"
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(GT_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
print(PRED_DIR)

for guidance_scale in [7.5, 3, 1]:
    for i, data in tqdm(enumerate(dataset), total=len(dataset)):

        # print("Check data", data.keys())
        # print("Check labels", len(data['label']), data["label"][0].shape, data['label'][0].min(), data['label'][0].max())
        # print("Check images", len(data['image']), data["image"][0].shape, data['image'][0].min(), data['image'][0].max())
        # print("Check paths", len(data['path']), data['path'][-1])
        # breakpoint()

        data_label = torch.stack(data['label']) if opt.mv else data['label']
        images = model.sample_images(data_label, h=256, w=256, num_steps=20, guidence_scale=guidance_scale)

        gt_image_paths = data["path"][-1]

        for j in range(len(images)):
            gt_image_path = gt_image_paths[j]
            image_name = gt_image_path.split('/')[-1].split('.')[0]

            # save the PIL image in guidance scale folder
            pred_sub_dir = os.path.join(PRED_DIR, str(guidance_scale))
            os.makedirs(pred_sub_dir, exist_ok=True)
            images[j].save(os.path.join(pred_sub_dir, image_name + '.jpg'))

            # copy the gt image to GT_DIR
            os.system(f"cp {gt_image_path} {GT_DIR}/{image_name}.jpg")

