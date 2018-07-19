
import random, sys, os, glob
import argparse, tqdm

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from fire import Fire

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import *
import transforms

from encoding import encode_binary
from models import DecodingNet
from logger import Logger

import IPython


logger = Logger("bits", ("orig", "rotate", "scale", "translate", "noise"),
                print_every=1, plot_every=5)

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


def test_transforms(model=None, image_files=VAL_FILES, name="test"):

    if model is None:
        model = nn.DataParallel(DecodingNet(distribution=transforms.encoding, n=64))

    if type(model) is str:
        x = nn.DataParallel(DecodingNet(distribution=transforms.encoding, n=64))
        x.module.load(model)
        model = x

    images = [im.load(image) for image in image_files]
    images = im.stack(images)
    targets = [binary.random(n=TARGET_SIZE) for _ in range(0, len(images))]
    model.eval()

    encoded_images = encode_binary(images, targets, model, n=96, verbose=True)

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


def evaluate(model, image, target):

    if type(model) is str:
        x = nn.DataParallel(DecodingNet(distribution=transforms.encoding, n=64))
        x.module.load(model)
        model = x

    image = im.torch(im.load(image)).unsqueeze(0)
    target = binary.parse(str(target))
    prediction = model(image).mean(dim=1).squeeze().cpu().data.numpy()
    prediction = binary.get(prediction)

    print (f"Target: {binary.str(target)}, Prediction: {binary.str(prediction)}, \
            Distance: {binary.distance(target, prediction)}")




if __name__ == "__main__":
    Fire()
