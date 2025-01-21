from tqdm import tqdm

import torch
import numpy as np
from torch import nn
from torchvision.transforms.v2 import GaussianNoise

from jnd import JND
from DiffJPEG.DiffJPEG import DiffJPEG

def jnra(img,
         target_img,
         model,
         alpha=1/255,
         steps=100,
         eps=32/255,
         clip_min=0.0,
         clip_max=1.0):
    device = img.device

    delta = torch.zeros_like(img, requires_grad=True).to(device)

    heatmap = JND(in_channels=3, out_channels=3).heatmaps(img.to('cpu')).to(device)

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * eps

    criterion = nn.MSELoss()

    preprocessings = [None, 25, 50, 75, 'GaussianNoise']

    gaussian_noise = GaussianNoise(sigma=0.05)

    for _ in tqdm(range(steps)):
        img_adv = img + delta

        delta_grads = []
        weights = []

        for preprocessing in preprocessings:
            delta.requires_grad_(True)
            if preprocessing is None:
                img_adv_proc = img_adv
            elif preprocessing == 'GaussianNoise':
                img_adv_proc = gaussian_noise(img_adv)
            else:
                diffjpeg = DiffJPEG(height=512, width=512, differentiable=True, quality=preprocessing).to(device)
                img_adv_proc = diffjpeg(img_adv)

            img_emb = model(img_adv_proc).latent_dist.sample()
            target_img_emb = model(target_img).latent_dist.sample()

            loss = criterion(img_emb, target_img_emb)

            loss.backward()

            delta_grads.append(delta.grad.sign().detach().clone())
            weights.append(loss.item())
            
            delta.grad.detach_()
            delta.grad.zero_()

        weights = torch.softmax(torch.Tensor(weights), 0)

        grad = torch.zeros_like(delta)

        for i in range(len(delta_grads)):
            grad += (weights[i] * delta_grads[i])
        
        delta.data = delta.data - alpha * grad
        delta.data[abs(delta.data) - heatmap > 0] = torch.sign(delta.data[abs(delta.data) - heatmap > 0]) * heatmap[abs(delta.data) - heatmap > 0]
        delta.data = torch.clamp(img + delta, clip_min, clip_max) - img 

        delta.grad.detach_()
        delta.grad.zero_()

    return img + delta