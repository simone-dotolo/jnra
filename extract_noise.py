import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
from skimage.io import imread

def extract_noise(data_path,
                  protection='glaze',
                  purification=None):
    '''
        Extract noise fingerprint.
    '''
    data_path = Path(data_path)
    
    # image_path -> data_path/[artist]/original/[image_name]
    image_paths = [path for path in data_path.glob('*/original/*') if str(path).endswith('jpg') or str(path).endswith('jpeg') or str(path).endswith('png')]

    noise_list = []

    for image_path in tqdm(image_paths):
        if purification:
            # protected_path -> data_path/[artist]/protected_purified/[protection_method]/[purification_method]/[image_name]
            protected_path = image_path.parent.parent / 'protected_purified' / protection / purification / image_path.name
        else:
            # protected_path -> data_path/[artist]/protected/[protection_method]/[image_name]
            protected_path = image_path.parent.parent / 'protected' / protection / image_path.name

        # [0, 255] -> [0.0, 1.0]
        img = np.float32(imread(image_path)) / 255.0
        img_adv = np.float32(imread(protected_path)) / 255.0

        # [-1.0 , 1.0]
        delta = img_adv - img

        noise_list.append(delta)

    noise_list = np.stack(noise_list, -1)

    if purification:
        output_name = protection + '_' + purification + '_noise_list.npy'
    else:
        output_name = protection + '_noise_list.npy'

    np.save(output_name, noise_list)

if __name__ == '__main__':
    # python3 extract_noise.py --data_path /home/simone.dotolo/style_mimicry/data/ --protection glaze --purification gaussian_noise
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', required=True, type=str, help='data path')
    parser.add_argument('--protection', required=True, type=str, help='protection\'s name', choices=['glaze', 'mist', 'advdm', 'photoguard', 'jnra'])
    parser.add_argument('--purification', default=None, type=str, help='purification\'s name', choices=['gaussian_noise', 'noisy_upscaling', 'resize', 'jpeg', 'diffpure'])

    args = parser.parse_args()

    extract_noise(data_path=args.data_path,
                  protection=args.protection,
                  purification=args.purification)