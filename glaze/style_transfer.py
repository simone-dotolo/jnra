import argparse
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler

def style_transfer(pipe_img2img,
                   target_style,
                   directory,
                   strength=0.4,
                   guidance=7.5,
                   diff_steps=50):
    
    dir_path = Path(directory)

    res_path = dir_path.parent / 'style_transfered'

    Path.mkdir(res_path, exist_ok=True)

    img_paths = [path for path in dir_path.glob('*') if str(path).endswith('jpg') or str(path).endswith('jpeg') or str(path).endswith('png')]

    for img_path in img_paths:
        prompt = f'{target_style} style'
        img = Image.open(img_path).convert('RGB')

        transf_img = pipe_img2img(prompt=prompt,
                                  image=img,
                                  strength=strength,
                                  guidance_scale=guidance,
                                  num_inference_steps=diff_steps).images[0]

        transf_img.save(res_path / img_path.name)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='stabilityai/stable-diffusion-2-1-base', type=str, help='style-transfer model')
    parser.add_argument('--dir', default='data/wikiart_edvard-munch/original', type=str, help='directory containing images to style-transfer')
    parser.add_argument('--target_style', default='Cubism by Picasso', type=str, help='target style')

    parser.add_argument('--strength', default=0.4, type=float)
    parser.add_argument('--guidance', default=7.5, type=float)
    parser.add_argument('--diff_steps', default=50, type=int)
    
    parser.add_argument('--manual_seed', default=42, type=int, help='manual seed')
    parser.add_argument('--device', default='cuda', type=str, help='device used')

    args = parser.parse_args()

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    torch.backends.cudnn.deterministic=True

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(args.model,
                                                                  revision='fp16',
                                                                  torch_dtype=torch.float16)
    
    pipe_img2img.scheduler = DPMSolverMultistepScheduler.from_config(pipe_img2img.scheduler.config)
    pipe_img2img = pipe_img2img.to(device)

    style_transfer(pipe_img2img=pipe_img2img,
                   target_style=args.target_style,
                   directory=args.dir,
                   strength=args.strength,
                   guidance=args.guidance,
                   diff_steps=args.diff_steps)
