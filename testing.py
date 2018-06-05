from __future__ import print_function
import IPython

import random, sys, os, glob, tqdm

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from utils import *
import transforms

from encoding import encode_binary
from models import DecodingNet

from scipy.ndimage import filters
from scipy import stats


# returns an image after a series of transformations
def p(x):
    x = transforms.resize_rect(x)
    x = transforms.rotate(transforms.scale(x, 0.6, 1.4), max_angle=30)
    x = transforms.gauss(x, min_sigma=0.8, max_sigma=1.2)
    x = transforms.translate(x)
    x = transforms.resize(x, rand_val=False, resize_val=224)
    return x

def sweep(images, targets, model, transform, min_val, max_val, samples=10, output_file="plot.jpg"):
    
    res_bin = []
    res_mse = []

    for val in tqdm.tqdm(np.linspace(min_val, max_val, samples), ncols=30):
        transformed = transform(images, val)
        predictions = model(transformed).mean(dim=1).cpu().data.numpy()

        mse_loss = np.mean([binary.mse_dist(x, y) for x, y in zip(predictions, targets)])
        binary_loss = np.mean([binary.distance(x, y) for x, y in zip(predictions, targets)])

        res_bin.append((val, binary_loss))
        res_mse.append((val, mse_loss))

    x, bits_off = zip(*res_bin)
    x, mse = zip(*res_mse)
    fig, ax1 = plt.subplots()
    ax1.plot(x, bits_off, 'b')
    ax1.set_ylim(0, TARGET_SIZE//2)
    ax1.set_ylabel('Number Incorrect Bits')
    ax2 = ax1.twinx()
    ax2.plot(x, mse, 'r')
    ax2.set_ylim(0, 0.25)
    ax2.set_ylabel('Mean Squared Error')
    plt.savefig(OUTPUT_DIR + output_file); 
    plt.cla()

def test_transforms(image_files=["images/house.png"], model=None):

    images = [im.load(image) for image in image_files]
    images = im.stack(images)
    targets = [binary.random(n=TARGET_SIZE) for _ in range(0, len(images))]

    if model == None: model = DecodingNet(distribution=p, n=64)
    model.eval()

    encoded_images = encode_binary(images, targets, model, verbose=True)
    
    sweep(encoded_images, targets, model,
            transform=lambda x, val: transforms.rotate(x, rand_val=False, theta=val),
            min_val=-0.6, max_val=0.6, samples=60,
            output_file="test_rotate.jpg")

    sweep(encoded_images, targets, model,
            transform=lambda x, val: transforms.scale(x, rand_val=False, scale_val=val),
            min_val=0.6, max_val=1.4, samples=60,
            output_file="test_scale.jpg") 

    # lambda x, val: transforms.noise(x, max_noise_val=val)

if __name__ == "__main__":
    test_transforms()



