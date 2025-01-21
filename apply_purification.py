import argparse
import io
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from diffusers import StableDiffusionUpscalePipeline, AutoPipelineForImage2Image
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.util import random_noise

def purify_resize(img, factor=4):
    h, w, _ = img.shape
    img_resized = resize(img, output_shape=(h // factor, w // factor))
    img_output = resize(img_resized, output_shape=(h, w))
    return img_output

def jpeg(img, qf=25):
    img *= 255.0
    img = np.uint8(img)
    buffered = io.BytesIO()
    imsave(buffered, img, format='jpeg', quality=qf)
    img_jpeg = imread(buffered)
    return img_jpeg

def gaussian_noise(img, std=0.05):
    var = std ** 2
    noisy_img = random_noise(img, mode='gaussian', var=var, clip=True)
    return noisy_img.clip(0, 1)

class DiffPure():
    '''
        DiffPure Class.
    '''

    def __init__(self,
                 model_id='stabilityai/stable-diffusion-xl-base-1.0',
                 device='cuda'):
        self.model_id = model_id
        self.device = device
        self.pipeline = AutoPipelineForImage2Image.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True).to(self.device)
    
    def __call__(self, img, prompt):
        return self.pipeline(prompt=prompt, strength=0.2, image=img, num_inference_steps=100)

class NoisyUpscaler():
    '''
        Noisy Upscaler Class.
    '''

    def __init__(self,
                 model_id='stabilityai/stable-diffusion-x4-upscaler',
                 device='cuda'):
        self.model_id = model_id
        self.device = device
        self.pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id).to(self.device)
    
    def __call__(self, img, prompt, gaussian_std=0.05, noise_level=320):
        img = gaussian_noise(img=img, std=gaussian_std)
        return self.pipeline(prompt=prompt, image=img, noise_level=noise_level)

def apply_purification(data_path,
                       purification_fn,
                       purification_name):
    '''
        Purify protected images.
    '''
    data_path = Path(data_path)

    # image_path -> data_path/[artist]/protected/[protection_method]/[image_name]
    image_paths = [path for path in data_path.glob('*/protected/*/*') if str(path).endswith('jpg') or str(path).endswith('jpeg') or str(path).endswith('png')]

    for image_path in tqdm(image_paths):
        # [protection_method]
        protection = image_path.parent.stem

        # output_dir -> data_path/[artist]/protected_purified/[protection_method]/[purification_method]
        output_dir = image_path.parent.parent.parent / 'protected_purified' / protection / purification_name

        output_dir.mkdir(exist_ok=True, parents=True)

        # output_path -> data_path/[artist]/protected_purified/[protection_method]/[purification_method]/[image_name]
        output_path = output_dir / image_path.name

        # [0, 255] -> [0.0, 1.0]
        img = np.float32(imread(image_path)) / 255.0

        if purification_name == 'noisy_upscaling' or purification_name == 'diffpure':
            metadata_path = image_path.parent.parent.parent / 'original' / 'metadata.csv'
            metadata = pd.read_csv(metadata_path)
            prompt = metadata.loc[metadata['file_name'] == image_path.name]['text']
            prompt = prompt.to_string(index=False)

            # Expand batch dimension
            img = np.expand_dims(img, axis=0)
            
            # Output [0.0, 255.0]
            purified_img = purification_fn(prompt=prompt,
                                           img=img).images[0]
        elif purification_name == 'jpeg':
            # Output [0, 255]
            purified_img = purification_fn(img)
        else:
            # Output [0.0, 1.0]
            purified_img = purification_fn(img)
            purified_img *= 255.0

        purified_img = np.uint8(purified_img)

        imsave(output_path, purified_img)

if __name__ == '__main__':
    # python3 apply_purification.py --data_path /home/simone.dotolo/style_mimicry/data/ --purification noisy_upscaling --device cuda
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', required=True, type=str, help='data path')
    parser.add_argument('--purification', required=True, type=str, help='purification\'s name', choices=['gaussian_noise', 'noisy_upscaling', 'resize', 'jpeg', 'diffpure'])
    parser.add_argument('--device', default='cpu', type=str, help='device to be used')

    args = parser.parse_args()

    if args.purification == 'noisy_upscaling':
        purification_fn = NoisyUpscaler(device=args.device)
    elif args.purification == 'diffpure':
        purification_fn = DiffPure(device=args.device)
    elif args.purification == 'gaussian_noise':
        purification_fn = gaussian_noise
    elif args.purification == 'resize':
        purification_fn = purify_resize
    elif args.purification == 'jpeg':
        purification_fn = jpeg

    apply_purification(data_path=args.data_path,
                       purification_fn=purification_fn,
                       purification_name=args.purification)