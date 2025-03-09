import os 
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Handle Datasets

class MVDataset(Dataset):
    def __init__(self, data, split='train', image_dir="data/sub_images", target_size=(224,224)):
        self.image_dir = image_dir
        self.target_size = target_size

        self.ids = data[split]
        self.labels = data['label'][self.ids]

        # resize images
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor()
        ])


        # print(len(self.data["train"]) + len(self.data["val"]) + len(self.data["test"]))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]

        # load and transform images
        image_path = os.path.join(self.image_dir, f"{id}.jpg")
        image = Image.open(image_path)

        if image.mode != 'RGB': # handle grayscale images
            image = image.convert('RGB') 

        image = self.transform(image)

        # assert image.shape==(3,224,224), f"The image {id} has a wrong shape of {image.shape}"

        return image, self.labels[idx]


class MVDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        data_path="data/Movies.pt",
        image_dir="data/sub_images",
        target_size=(224,224),
        batch_size=32
    ):
        super().__init__()

        self.data = torch.load(data_path, weights_only=False)
        self.image_dir = image_dir
        self.target_size=target_size

        self.batch_size = batch_size
    
    def setup(self, stage=None):
        self.train_dataset = MVDataset(self.data, 'train', self.image_dir, self.target_size)
        self.val_dataset = MVDataset(self.data, 'val', self.image_dir, self.target_size)
        self.test_dataset = MVDataset(self.data, 'test', self.image_dir, self.target_size)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataloader, batch_size=self.batch_size)
    

# Start writing the models

class FCNN(nn.Module):
    def __init__(self, input_size, output_size=20):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.model(x.view(x.size(0), -1)) # flatten input to 1d


class CNN(nn.Module):
    def __init__(self, h, w, output_size=20):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64*(h//4)*(w//4), 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
    
    def forward(self, x):
        return self.model(x)
    

class LSTM(nn.Module):
    def __init__(self, output_size=20, patch_size=8, image_size=224):
        super().__init__()

        self.patch_size = patch_size,
        self.image_size = image_size,
        self.num_patches = (image_size // patch_size) ** 2

        self.lstm = nn.LSTM(patch_size*patch_size*3, 128, num_layers=6, batch_first=True)

        self.fc = nn.Linear(128, output_size)
    
    def patchify(self, x):
        b, c, h, w = x.shape
        assert h==w and h==self.image_size, f"The input image must be of shape {self.image_size} x {self.image_size}"
        
        patches = x.reshape(b, c, self.num_patches, self.patch_size, self.num_patches, self.patch_size)

        patches = patches.permute(0, 2, 4, 1, 3, 5) # to be b, nh, nw, c, ph, pw
        patches = patches.reshape(b, self.num_patches*self.num_patches, c*self.patch_size*self.patch_size) # to be b, m, n

        return patches

    def forward(self, x):

        patches = self.patchify(x)
        
        _, (h_n, c_n) = self.lstm(patches)
        
        out = self.fc(h_n[-1])
        return out


class MVModel(pl.LightningModule):
    def __init__(self, model_type, image_size=(224, 224), lr=1e-3):
        super().__init__()

        h, w = image_size
        num_classes = 20

        if model_type == "fcnn":
            self.model = FCNN(h*w*3, num_classes)
        
        elif model_type == "cnn":
            self.model = CNN(h, w, num_classes)

        elif model_type == "lstm":
            self.model = LSTM(num_classes, 8, h)

        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x) 

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) 
        loss = self.criterion(y_hat, y) 
        return loss 

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        # evaluate teh accuracy
        acc = (y_hat.argmax(dim=-1) == y).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        acc = (y_hat.argmax(dim=-1) == y).float().mean()
        self.log("test_acc", acc, on_epoch=True)
        return acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["fcnn", "cnn", "lstm"], required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    args = parser.parse_args()

    # set the device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    datamodule = MVDataModule(        
        data_path="data/Movies.pt",
        image_dir="data/sub_images",
        target_size=(224,224),
        batch_size=args.batch_size
    )

    model = MVModel(
        model_type=args.model_type,
        image_size=(224, 224),
        lr = args.learning_rate
    )


    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/{args.model_type}-e{args.epochs}-bs{args.batch_size}-lr{args.learning_rate}',
        filename='{epoch}-{val_acc:.2f}',
        monitor='val_acc',
        mode='max',
        save_top_k=3, 
    )

    trainer = Trainer(
        max_epochs=30,
        accelerator=device,
        devices=1,
        precision=16,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=-1,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, datamodule=datamodule)
    
    