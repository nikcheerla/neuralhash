
import random, sys, os, glob
import argparse, tqdm

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import *
import transforms

from encoding import encode_binary
from models import DecodingNet
from logger import Logger


logger = Logger("bits", ("orig", "rotate", "scale", "translate", "noise"),
                print_every=1, plot_every=5)

# returns an image after a series of transformations
def p(x):
    x = transforms.noise(x, intensity=0.08)
    x = transforms.resize_rect(x)
    x = transforms.rotate(transforms.scale(x, 0.6, 1.4), max_angle=30)
    x = transforms.gauss(x, min_sigma=0.8, max_sigma=1.2)
    x = transforms.translate(x)
    x = transforms.resize(x, rand_val=False, resize_val=224)
    return x

def sweep(images, targets, model, transform, \
            name, transform_name, 
            min_val, max_val, samples=10):
    
    results = []
    for val in tqdm.tqdm(np.linspace(min_val, max_val, samples), ncols=30):
        transformed = transform(images, val)
        predictions = model(transformed).mean(dim=1).cpu().data.numpy()

        mse_loss = np.mean([binary.mse_dist(x, y) for x, y in zip(predictions, targets)])
        binary_loss = np.mean([binary.distance(x, y) for x, y in zip(predictions, targets)])
        results.append((val, binary_loss, mse_loss))

    x, bits_off, mse = zip(*results)

    print (transform_name, np.mean(bits_off))
    logger.step(transform_name, np.mean(bits_off))

    np.savez_compressed(f"{OUTPUT_DIR}/{name}_{transform_name}.npz", 
                        x=x, bits_off=bits_off, mse=mse)

    fig, ax1 = plt.subplots()
    ax1.plot(x, bits_off, 'b')
    ax1.set_ylim(0, TARGET_SIZE//2)
    ax1.set_ylabel('Number Incorrect Bits')
    ax2 = ax1.twinx()
    ax2.plot(x, mse, 'r')
    ax2.set_ylim(0, 0.25)
    ax2.set_ylabel('Mean Squared Error')
    plt.savefig(f"{OUTPUT_DIR}/{name}_{transform_name}.jpg"); 
    plt.cla(); plt.clf(); plt.close()

def test_transforms(model=None, image_files=VAL_FILES, name="iter"):

    images = [im.load(image) for image in image_files]
    images = im.stack(images)
    targets = [binary.random(n=TARGET_SIZE) for _ in range(0, len(images))]
    model.eval()

    encoded_images = encode_binary(images, targets, model, verbose=True)

    for img, encoded_im, filename, target in zip(images, encoded_images, image_files, targets):
        im.save(im.numpy(img), file=f"{OUTPUT_DIR}{binary.str(target)}_original_{filename.split('/')[-1]}")
        im.save(im.numpy(encoded_im), file=f"{OUTPUT_DIR}{binary.str(target)}_encoded_{filename.split('/')[-1]}")

    predictions = model(encoded_images).mean(dim=1).cpu().data.numpy()
    binary_loss = np.mean([binary.distance(x, y) for x, y in zip(predictions, targets)])

    logger.step("orig", binary_loss)

    sweep(encoded_images, targets, model,
            transform=lambda x, val: transforms.rotate(x, rand_val=False, theta=val),
            name=name, transform_name="rotate",
            min_val=-0.6, max_val=0.6, samples=60)

    sweep(encoded_images, targets, model,
            transform=lambda x, val: transforms.scale(x, rand_val=False, scale_val=val),
            name=name, transform_name="scale",
            min_val=0.6, max_val=1.4, samples=50) 

    sweep(encoded_images, targets, model,
            transform=lambda x, val: transforms.translate(x, max_val=val),
            name=name, transform_name="translate",
            min_val=0.0, max_val=0.3, samples=10)

    sweep(encoded_images, targets, model,
            transform=lambda x, val: transforms.noise(x, intensity=val),
            name=name, transform_name="noise",
            min_val=0.0, max_val=0.1, samples=15)
    model.train()

def test_encoded_imgs(model, image_files, targets, name='test'):
    
    encoded_images = [im.load(image) for image in image_files]
    encoded_images = im.stack(encoded_images)

    predictions = model(encoded_images).mean(dim=1).cpu().data.numpy()
    # score_targets = binary.target(targets).unsqueeze(1).expand_as(predictions)

    binary_loss = np.mean([binary.distance(x, y) for x, y in zip(predictions, targets)])
    mse_loss = np.mean([binary.mse_dist(x, y) for x, y in zip(predictions, targets)])
    print("mse_loss = ", mse_loss)
    print("binary_loss = ", binary_loss)
    # bce_loss = F.binary_cross_entropy(predictions, score_targets)
    # print("bce loss = ", bce_loss)
    # logger.step("orig", binary_loss)

    sweep(encoded_images, targets, model,
            transform=lambda x, val: transforms.rotate(x, rand_val=False, theta=val),
            name=name, transform_name="rotate",
            min_val=-0.6, max_val=0.6, samples=60)

    sweep(encoded_images, targets, model,
            transform=lambda x, val: transforms.scale(x, rand_val=False, scale_val=val),
            name=name, transform_name="scale",
            min_val=0.6, max_val=1.4, samples=50) 

    sweep(encoded_images, targets, model,
            transform=lambda x, val: transforms.translate(x, max_val=val),
            name=name, transform_name="translate",
            min_val=0.0, max_val=0.3, samples=10)

    sweep(encoded_images, targets, model,
            transform=lambda x, val: transforms.noise(x, intensity=val),
            name=name, transform_name="noise",
            min_val=0.0, max_val=0.05, samples=10)
    model.train()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Encodes a set of images and benchmarks robustness.')
    parser.add_argument('--path', default=None, help="Path of model to load")
    args = parser.parse_args()

    model = nn.DataParallel(DecodingNet(distribution=p, n=96))
    if args.path is not None: model.module.load(args.path)
    model.eval()

    # test_transforms(model, image_files=['images/starry_night_orig.jpg'], name="starry_night")
    # test_transforms(model, image_files=VAL_FILES[16:32], name="test")
    test_encoded_imgs(model, ['images/starry_test3.png'], [binary.parse('01100110100000101001011111110010')], name='starry_test')



