import argparse

import numpy as np
from matplotlib import pyplot as plt

def visualize_noise(noise_path,
                    protection):
    rey = np.load(noise_path)

    rey = np.abs(np.fft.fftshift(np.fft.fftn(rey, axes=(0, 1)), axes=(0, 1))) ** 2

    rey = np.mean(rey,(-1, -2))

    rey = np.log(rey + 1)

    out_name = protection + '_noise.png'

    plt.imsave(out_name, rey, vmin=rey.min(), vmax=rey.max(), cmap='inferno')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--noise_path', required=True, type=str, help='noise path')
    parser.add_argument('--protection', required=True, type=str, help='protection\'s name')

    args = parser.parse_args()

    visualize_noise(args.noise_path,
                    args.protection)