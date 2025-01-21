import argparse
from pathlib import Path
from tqdm import tqdm

import lpips
import torch
import numpy as np
from skimage.io import imread
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

def evaluate_similarity(original_path,
                        protected_path,
                        device):

    lpips_sim = lpips.LPIPS(net='vgg').to(device)

    original_image_paths = [path for path in Path(original_path).glob('*') if str(path).endswith('jpg') or str(path).endswith('jpeg') or str(path).endswith('png')]
    protected_image_paths = [path for path in Path(protected_path).glob('*') if str(path).endswith('jpg') or str(path).endswith('jpeg') or str(path).endswith('png')]

    cum_lpips = 0
    cum_ssim = 0
    cum_psnr = 0

    for i in tqdm(range(len(original_image_paths))):
        original_image_path = original_image_paths[i]
        protected_image_path = protected_image_paths[i]

        original_image = np.float32(imread(original_image_path)) / 255.0
        protected_image = np.float32(imread(protected_image_path)) / 255.0

        original_lpips_input = torch.from_numpy(original_image * 2.0 - 1.0).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        protected_lpips_input = torch.from_numpy(protected_image * 2.0 - 1.0).unsqueeze(0).permute(0, 3, 1, 2).to(device)

        cum_lpips += lpips_sim(original_lpips_input, protected_lpips_input).item()
        cum_ssim += structural_similarity(original_image, protected_image, data_range=1.0, channel_axis=2, gaussian_weights=True)
        cum_psnr += peak_signal_noise_ratio(original_image, protected_image)
    
    cum_lpips /= len(original_image_paths)
    cum_ssim /= len(original_image_paths)
    cum_psnr /= len(original_image_paths)

    print(f'LPIPS: {cum_lpips}')    
    print(f'SSIM: {cum_ssim}')
    print(f'PSNR: {cum_psnr}')

if __name__ == '__main__':
    # python3 protection_invasiveness.py --original_path data/wikiart_rene-magritte/original/ --protected_path data/wikiart_rene-magritte/protected/photoguard --device cuda:0
    parser = argparse.ArgumentParser()

    parser.add_argument('--original_path', required=True, type=str, help='original images path')
    parser.add_argument('--protected_path', required=True, type=str, help='protected images path')
    parser.add_argument('--device', default='cpu', type=str, help='device to be used')

    args = parser.parse_args()

    evaluate_similarity(original_path=args.original_path,
                        protected_path=args.protected_path,
                        device=args.device)