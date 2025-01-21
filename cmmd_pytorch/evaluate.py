import argparse
from pathlib import Path

import torch
import piq
import numpy as np
from skimage.io import imread
from main import compute_cmmd

if __name__ == '__main__':
    # python3 evaluate.py --original_path /home/simone.dotolo/style_mimicry/generated_images/wikiart_rene-magritte/original --generated_path /home/simone.dotolo/style_mimicry/generated_images/wikiart_rene-magritte/protected/photoguard --device cuda:3
    parser = argparse.ArgumentParser()

    parser.add_argument('--original_path', required=True, type=str, help='original images path')
    parser.add_argument('--generated_path', required=True, type=str, help='generated images path')
    parser.add_argument('--device', default='cpu', type=str, help='device to be used')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--dims', default=2048, type=int, help='dimensionality of Inception features to use')
    parser.add_argument('--num_workers', default=1, type=int, help='number of processes to use for data loading')

    args = parser.parse_args()

    generated_path = args.generated_path
    original_path = args.original_path

    image_paths = [path for path in Path(generated_path).glob('*') if str(path).endswith('jpg') or str(path).endswith('jpeg') or str(path).endswith('png')]

    clip_scores = []

    for image_path in image_paths:
        img = np.float32(imread(image_path)) / 255.0

        img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).to(args.device)

        clip_iqa_index = piq.CLIPIQA(data_range=1.0).to(args.device)(img)

        clip_scores.append(clip_iqa_index.item())

    clip_iqa_score = sum(clip_scores) / len(clip_scores)
    cmmd_score = compute_cmmd(generated_path, original_path, None, args.batch_size)

    print(f'Clip-IQA: {clip_iqa_score:.4f}')
    print(f'CMMD score: {cmmd_score:.4f}')