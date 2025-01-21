import argparse
from pathlib import Path

from matplotlib import pyplot as plt
from skimage.io import imread

protections = ['glaze', 'mist', 'photoguard', 'advdm', 'jnra']
purifications = ['jpeg', 'gaussian_noise', 'resize', 'diffpure', 'noisy_upscaling']

def visualize_results(data_path, artist, img_name):
    data_path = Path(data_path)
    
    n_rows = len(protections)
    n_cols = len(purifications) + 1

    plt.figure(figsize=(20, 20))

    for i in range(n_rows * n_cols):

        plt.subplot(n_rows, n_cols, i + 1)

        protection = protections[i // n_cols]
        
        if i % n_cols != 0:
            # Purification
            purification = purifications[i % n_cols - 1]

            img_path = data_path / artist / 'protected_purified' / protection / purification / img_name
            title = f'{protection.capitalize()} + {purification.capitalize()}'
        else:
            # No purification
            img_path = data_path / artist / 'protected' / protection / img_name
            title = f'{protection.capitalize()}'
        
        img = imread(img_path)

        plt.imshow(img)
        plt.title(title, fontsize=20)
        plt.axis(False)

    plt.tight_layout()
    img_name = img_name.split('.')[0]
    plt.savefig(f'finetuning_results_{img_name}.png')

if __name__ == '__main__':
    # python3 display_results.py --data_path /home/simone.dotolo/style_mimicry/generated_images/ --artist wikiart_rene-magritte --img 0
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', required=True, type=str, help='data path')
    parser.add_argument('--artist', required=True, type=str, help='artist\'s name', choices=['wikiart_rene-magritte', 'wikiart_eugene-bodin'])
    parser.add_argument('--img', default=0, type=int, help='image number [0-17]', choices=range(0, 18))

    args = parser.parse_args()

    img_name = str(args.img).zfill(4) + '.png'

    visualize_results(args.data_path, args.artist, img_name)