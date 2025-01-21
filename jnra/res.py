from pathlib import Path
from tqdm import tqdm

import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize

factor = 4

for img_path in Path('train').glob('*.png'):
    img = np.float32(imread(img_path)) / 255.0
    h, w, _ = img.shape
    img_resized = resize(img, output_shape=(h // factor, w // factor))
    img_resized *= 255.0
    img_resized = np.uint8(img_resized)
    imsave(img_path, img_resized)

for img_path in Path('val').glob('*.png'):
    img = np.float32(imread(img_path)) / 255.0
    h, w, _ = img.shape
    img_resized = resize(img, output_shape=(h // factor, w // factor))
    img_resized *= 255.0
    img_resized = np.uint8(img_resized)
    imsave(img_path, img_resized)
