import torch
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn
import lightning as L
import wandb
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
import numpy as np

from .modules import GCN, CLIPAdapter


class GC_Diffusion(L.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.pretrained_model_name_or_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        self.clip_model_name_or_path = "openai/clip-vit-large-patch14"

        self.gcn = GCN(77, 77, num_layers=2)
        self.clip_adapter = CLIPAdapter()
        # self.vae = AutoencoderKL.from_pretrained(
        #     self.pretrained_model_name_or_path,
        #     subfolder="vae",
        # )
        # self.unet = UNet2DConditionModel.from_pretrained(
        #     self.pretrained_model_name_or_path,
        #     subfolder="unet",
        # )
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.pretrained_model_name_or_path,
            safety_checker=None,
        )

        self.gen_pipeline = StableDiffusionPipeline.from_pretrained(
                self.pretrained_model_name_or_path,
                safety_checker=None,
            )

        # we fix the VAE and UNet parameters
        self.vae = pipeline.vae
        self.unet = pipeline.unet

        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name_or_path)
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name_or_path)

        # # we fix the VAE and UNet parameters
        # for param in self.vae.parameters():
        #     param.requires_grad = False
        # for param in self.unet.parameters():
        #     param.requires_grad = False
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)

        self.scheduler = diffusers.schedulers.DDPMScheduler()
    
    def embed_clip(self, labels):
        """
        labels should be of shape b, c, h, w
        """

        # map from range [-1, 1] to [0, 1]
        labels = (labels + 1) / 2

        inputs = self.clip_processor(images=labels, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            embs = self.clip_model.get_image_features(**inputs)
        return embs


    def batch_forward(self, labels):
        # labels: [7, batch_size, 3, 256, 256]
        n_imgs, batch_size = labels.shape[0], labels.shape[1]

        # we replace the last label with the noisy latent
        # labels[-1] = noisy_latents

        # we replace the last label with a random noise
        vmin = labels.min()
        vmax = labels.max()
        labels[-1] = torch.rand_like(labels[-1], device=self.device) * (vmax - vmin) + vmin
        

        # "0_0.jpg", "0_1.jpg", "1_0.jpg", "1_1.jpg", "0.jpg", "1.jpg", "z.jpg"
        adj = torch.tensor([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 1, 0],
            [1, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 1, 1]
        ], dtype=labels.dtype, device=self.device)
        # repeat adj over batch
        adj = adj.repeat(batch_size, 1, 1)

        # we also need to encode label (images) with vae
        # we first turn a n, b, c, h, w tensor into a n*b, c, h, w tensor
        h = labels.view(-1, *labels.shape[2:])
        # h = self.vae.encode(h).latent_dist.mode()
        h = self.embed_clip(h)
        h = h.view(n_imgs, batch_size, *h.shape[1:])
        # turn n, b, c, h, w into b, n, c, h, w
        # print("!!!check h shape before", h.shape)
        h = h.permute(1, 0, 2)

        h = h.unsqueeze(2).repeat(1, 1, 77, 1)
# 
        gcn_embs = self.gcn(h, adj)
        # print("!!!check gcn_embs shape", gcn_embs.shape)

        gcn_embs = gcn_embs.permute(1, 0, 2, 3)
        gcn_hidden_states = gcn_embs[-1]
        gcn_hidden_states = self.clip_adapter(gcn_hidden_states)
        # print("!!!Shape check for GCN input", h.shape)
        # print("!!!Shape check for GCN output", gcn_embs.shape)
        # print("!!!Shape check for GCN adapter output", gcn_hidden_states.shape)

        return gcn_hidden_states

    def training_step(self, batch, batch_idx):
        labels = batch["label"]
        images = batch["image"]

        labels = torch.stack(labels, dim=0)
        
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        steps = steps.long()
        noisy_latents = self.scheduler.add_noise(latents, noise, steps)

        gcn_hidden_states = self.batch_forward(labels)

        model_pred = self.unet(noisy_latents, steps, encoder_hidden_states=gcn_hidden_states, return_dict=False)[0]
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction='mean')

        self.log("train_loss", loss)
        return loss


    def sample_images(self, labels, h=256, w=256, num_steps=50, guidence_scale=3):
        batch_size = labels.shape[1]
        # we start with a batch of random noise
        shape = (batch_size, 3, h // 8, w // 8)

        # pipeline = StableDiffusionPipeline.from_pretrained(
        #     self.pretrained_model_name_or_path,
        #     safety_checker=None,
        # )

        gcn_hidden_states = self.batch_forward(labels)

        images = self.gen_pipeline(
            height=h, width=w,
            num_inference_steps=num_steps,
            guidance_scale=guidence_scale,
            prompt_embeds=gcn_hidden_states,
        ).images

        return images

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['label']

        labels = torch.stack(labels, dim=0)
        # print("!!! check labels attr", type(labels), len(labels), type(labels[0]))
        # print("!!! check images attr", type(images), len(images), type(images[0]))

        image_samples = self.sample_images(labels)

        # turn a list of PIL images into tensors
        images = (images + 1) / 2.0
        images = images.clamp(0, 1).cpu()
        gt_images = []
        for img in images:
            img = img.permute(1, 2, 0)
            arr = (img * 255).numpy().astype(np.uint8)
            gt_images.append(Image.fromarray(arr))
        
        
        # put two images side by side
        log_images = []
        for i, img in enumerate(gt_images):
            gen_img = image_samples[i]
            new_width = img.width + gen_img.width
            new_height = img.height
            new_img = Image.new('RGB', (new_width, new_height))  # or 'RGBA' if using transparency

            # Paste the two images
            new_img.paste(img, (0, 0))
            new_img.paste(gen_img, (img.width, 0))
            log_images.append(new_img)
        # image_grid = make_image_grid(log_images, images)


        # image_samples = (image_samples + 1) / 2.0
        
        # # put each image with each image_sample in a grid
        # image_grid = make_image_grid(image_samples, images)

        if batch_idx == 0:
            # self.logger.experiment.add_image('image_samples', image_grid, global_step=self.global_step)
            # log image with wandb
            # self.logger.experiment.log({"image_samples": wandb.Image(image_grid)})

            # log the list of PIL images with wandb
            self.logger.experiment.log({"image_samples": [wandb.Image(image) for image in log_images]})
        

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.opt.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]