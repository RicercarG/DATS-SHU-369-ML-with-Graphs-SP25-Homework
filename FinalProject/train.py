import os

import numpy as np
import torch
import torchvision
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import GC_Diffusion

os.environ["WANDB_MODE"] = "offline"

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

model = GC_Diffusion(opt)

# for i, data in enumerate(dataset):
#     # print(data)
#     print(type(data))
#     print(data.keys())
#     print("label", len(data["label"]), data["label"][0].shape)
#     print("image", len(data["image"]), data["image"][0].shape)

#     # save a sample of the labels as image
#     for j in range(len(data["label"])):
#         img = data["label"][j][1]
#         print(type(img), img.shape, img.min(), img.max())

#         img = (img + 1) / 2
#         torchvision.utils.save_image(img, "label_" + str(i) + "_" + str(j) + ".png")

#     break

# save the last 3 checkpoints
checkpoint_callback = ModelCheckpoint(
    dirpath=opt.checkpoints_dir,
    filename="checkpoint-{epoch}",
    save_last=True,
    save_top_k=3,
    monitor="train_loss",
    mode="min",
)

wandb_logger = WandbLogger(project="gnn-final")
trainer = L.Trainer(
    max_epochs=opt.niter, 
    logger=wandb_logger,
    callbacks=[checkpoint_callback], 
    accelerator="gpu", 
    devices=1,
    limit_val_batches=1,
    )

trainer.fit(model, train_dataloaders=dataset, val_dataloaders=dataset)