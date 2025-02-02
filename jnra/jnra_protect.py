from pathlib import Path
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from matplotlib import pyplot as plt
from skimage.io import imsave
import numpy as np
from PIL import Image
import torch

from jnra import jnra

data_path = Path('../data/wikiart_zdzislaw-beksinki/original')

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    pipe_img2img = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1-base', torch_dtype=torch.float32)
    pipe_img2img.scheduler = DPMSolverMultistepScheduler.from_config(pipe_img2img.scheduler.config)
    pipe_img2img = pipe_img2img.to(device)
    for name, param in pipe_img2img.vae.encoder.named_parameters():
        param.requires_grad = False

    image_paths = [path for path in data_path.glob('*') if str(path).endswith('jpg') or str(path).endswith('jpeg') or str(path).endswith('png')]

    for image_path in tqdm(image_paths):

        output_dir = image_path.parent.parent / 'protected' / 'jnra'

        output_dir.mkdir(exist_ok=True, parents=True)

        output_path = output_dir / image_path.name

        target_path = 'MIST.png'

        img = Image.open(image_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB').resize((512, 512))

        img = np.float32(img) / 255.0

        img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)

        target_img = np.float32(target_img) / 255.0

        target_img = torch.from_numpy(target_img).unsqueeze(0).permute(0, 3, 1, 2)

        img, target_img = img.to(device), target_img.to(device)

        img_adv = jnra(img=img,
                       target_img=target_img,
                       model=pipe_img2img.vae.encode,
                       alpha=2/255,
                       steps=100)
                
        img_adv = img_adv.permute(0, 2, 3, 1)[0].cpu().detach().numpy()

        img_adv *= 255.0

        img_adv = np.uint8(img_adv)

        imsave(output_path, img_adv)
    
if __name__ == '__main__':
    main()
