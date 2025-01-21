from tqdm import tqdm

import numpy as np
import lpips
import torch
from torch import nn

def glaze(img,
          transf_img,
          model,
          p=0.1,
          alpha=0.1,
          iters=500,
          lr=0.002):

    device = img.device

    delta = (torch.rand(img.shape) * 2 * p - p).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam([delta], lr=lr)

    loss_fn_alex = lpips.LPIPS(net='vgg').to(device)

    pbar = tqdm(range(iters))

    delta.requires_grad_(True)
    
    for _ in pbar:
        img_adv = img + delta
        img_adv.data = torch.clamp(img_adv, min=-1.0, max=1.0)

        img_emb = model(img_adv).latent_dist.sample()
        transf_img_emb = model(transf_img).latent_dist.sample()

        optimizer.zero_grad()

        d = loss_fn_alex(img, img_adv)
        sim_loss = alpha * max(d - p, 0)
        loss = criterion(img_emb, transf_img_emb) + sim_loss

        loss.backward()
        optimizer.step()

        pbar.set_description(f"[Running glaze]: Loss {loss.item():.5f} | sim loss {alpha * max(d.item() - p, 0):.5f} | dist {d.item():.5f}")
    
    img_adv = img + delta

    img_adv.data = torch.clamp(img_adv, min=-1.0, max=1.0)

    return img_adv, delta