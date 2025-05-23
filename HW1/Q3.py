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
        self.target_size = target_size

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
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
    

# Start writing the models

class FCNN(nn.Module):
    def __init__(self, input_size, output_size=20):
        super().__init__()

        layers = []
        in_features = input_size
        hidden_features = [1024, 512, 128]

        for f in hidden_features:
            layers.append(nn.Linear(in_features, f))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(f)) # add some tricks
            layers.append(nn.Dropout(0.2))
            in_features = f 

        layers.append(nn.Linear(in_features, output_size))

        self.model = nn.Sequential(*layers) 

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
    def __init__(self, output_size=20, patch_size=8):
        super().__init__()

        self.patch_size = patch_size # for square patches only

        self.lstm = nn.LSTM(patch_size*patch_size*3, 1024, num_layers=6, batch_first=True)
        self.fc = nn.Linear(1024, output_size)
    
    def patchify(self, x):
        b, c, h, w = x.shape
        p = self.patch_size
        nh = h // p
        nw = w // p

        patches = x.reshape(b, c, nh, p, nw, p)

        patches = patches.permute(0, 2, 4, 1, 3, 5) # to be b, nh, nw, c, p, p
        patches = patches.reshape(b, nh*nw, c*p*p) # to be b, m, n

        return patches

    def forward(self, x):

        patches = self.patchify(x)
        
        _, (h_n, c_n) = self.lstm(patches)
        
        out = self.fc(h_n[-1])
        return out


class MVModel(pl.LightningModule):
    def __init__(self, model_type, image_size=(224, 224), lr=1e-3, weight_decay=1e-5):
        super().__init__()

        h, w = image_size
        num_classes = 20

        if model_type == "fcnn":
            self.model = FCNN(h*w*3, num_classes)
        
        elif model_type == "cnn":
            self.model = CNN(h, w, num_classes)

        elif model_type == "lstm":
            self.model = LSTM(num_classes, 8)

        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay

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
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["fcnn", "cnn", "lstm"], required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

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
        save_top_k=1, 
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator=device,
        devices=1,
        precision=16,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=-1,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, datamodule=datamodule)

    print(f"Start testing for model {args.model_type} with lr {args.learning_rate} and weight decay {args.weight_decay}")
    # Load the best checkpoint after training
    best_model_path = checkpoint_callback.best_model_path
    best_model = MVModel.load_from_checkpoint(best_model_path, model_type=args.model_type)
    trainer.test(best_model, datamodule=datamodule)
    
    