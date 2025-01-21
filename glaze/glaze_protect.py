from pathlib import Path
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from matplotlib import pyplot as plt
from skimage.io import imsave
import numpy as np
from PIL import Image
import torch

from glaze import glaze

data_path = Path('/home/simone.dotolo/style_mimicry/data')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pipe_img2img = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1-base', torch_dtype=torch.float32)
    pipe_img2img.scheduler = DPMSolverMultistepScheduler.from_config(pipe_img2img.scheduler.config)
    pipe_img2img = pipe_img2img.to(device)
    for name, param in pipe_img2img.vae.encoder.named_parameters():
        param.requires_grad = False

    # image_path -> data_path/[artist]/original/[image_name]
    image_paths = [path for path in data_path.glob('*/original/*') if str(path).endswith('jpg') or str(path).endswith('jpeg') or str(path).endswith('png')]

    list_noise = []

    for image_path in tqdm(image_paths):

        # output_dir -> data_path/[artist]/protected/glaze
        output_dir = image_path.parent.parent / 'protected' / 'glaze'

        output_dir.mkdir(exist_ok=True, parents=True)

        # output_dir -> data_path/[artist]/protected/glaze/[image_name]
        output_path = output_dir / image_path.name

        img = Image.open(image_path).convert('RGB')
        transf_img = Image.open(image_path.parent.parent / 'style_transfered' / image_path.name).convert('RGB')

        # [0, 255] -> [-1.0, 1.0]
        img = np.float32(img) / 255.0
        img = img * 2.0 - 1.0
        # (h, w, c) -> (b, c, h, w)
        img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)

        # [0, 255] -> [-1.0, 1.0]
        transf_img = np.float32(transf_img) / 255.0
        transf_img = transf_img * 2.0 - 1.0
        # (h, w, c) -> (b, c, h, w)
        transf_img = torch.from_numpy(transf_img).unsqueeze(0).permute(0, 3, 1, 2)

        img, transf_img = img.to(device), transf_img.to(device)

        img_adv, delta = glaze(img,
                               transf_img,
                               pipe_img2img.vae.encode,
                               iters=500)
        
        delta = delta.permute(0, 2, 3, 1)[0].cpu().detach().numpy()

        img_adv = img_adv.permute(0, 2, 3, 1)[0].cpu().detach().numpy()

        img_adv = img_adv / 2.0 + 0.5

        img_adv *= 255.0

        img_adv = np.uint8(img_adv)

        imsave(output_path, img_adv)

        list_noise.append(delta)
    
    rey = np.stack(list_noise, -1)

    np.save('glaze_noise_list.npy', rey)

    print(type(rey), rey.shape)

    rey = np.abs(np.fft.fftshift(np.fft.fftn(rey, axes=(0, 1)), axes=(0, 1))) ** 2
    rey = np.mean(rey,(-1, -2))
    energy2 = np.mean(rey)
    rey = rey / 4 / energy2

    plt.imsave('glaze_noise.png',  rey.clip(0, 1), vmin=0, vmax=1)
    plt.imshow(rey.clip(0, 1), clim=[0, 1], extent=[-0.5, 0.5, 0.5, -0.5])
    
if __name__ == '__main__':
    main()