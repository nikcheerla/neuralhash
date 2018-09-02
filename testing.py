
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
from models import DecodingNet, DecodingGramNet
from logger import Logger

import IPython

def loss_func(model, x, targets):
    scores = model.forward(x)
    predictions = scores.mean(dim=1)
    score_targets = binary.target(targets).unsqueeze(1).expand_as(scores)

    return F.binary_cross_entropy(scores, score_targets), \
        predictions.cpu().data.numpy().round(2)

logger = Logger("bits", ("orig", "rotate", "scale", "translate", "noise", "crop"),
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


def test_transforms(model=None, image_files=VAL_FILES[:8], name="test", max_iter=350):

    if isinstance(model, str):
        model = nn.DataParallel(DecodingGramNet.load(distribution=transforms.encoding,
                                            n=96, weights_file=model))

    images = [im.load(image) for image in image_files]
    images = im.stack(images)
    targets = [binary.random(n=TARGET_SIZE) for _ in range(0, len(images))]
    model.eval()

    encoded_images = encode_binary(images, targets, model, n=ENCODING_DIST_SIZE, verbose=True, max_iter=max_iter, use_weighting=True)

    for img, encoded_im, filename, target in zip(images, encoded_images, image_files, targets):
        im.save(im.numpy(img), file=f"{OUTPUT_DIR}_{binary.str(target)}_original_{filename.split('/')[-1]}")
        im.save(im.numpy(encoded_im), file=f"{OUTPUT_DIR}_{binary.str(target)}_encoded_{filename.split('/')[-1]}")

    model.module.set_distribution(transforms.identity, n=1)
    predictions = model(encoded_images).mean(dim=1).cpu().data.numpy()
    binary_loss = np.mean([binary.distance(x, y) for x, y in zip(predictions, targets)])

    logger.step("orig", binary_loss)
    sweep(encoded_images, targets, model,
            transform=lambda x, val: transforms.rotate(x, rand_val=False, theta=val),
            name=name, transform_name="rotate",
            min_val=-0.6, max_val=0.6, samples=80)

    sweep(encoded_images, targets, model,
            transform=lambda x, val: transforms.scale(x, rand_val=False, scale_val=val),
            name=name, transform_name="scale",
            min_val=0.6, max_val=1.4, samples=50) 

    sweep(encoded_images, targets, model,
            transform=lambda x, val: transforms.translate(x, max_val=val),
            name=name, transform_name="translate",
            min_val=0.0, max_val=1.0, samples=50)

    sweep(encoded_images, targets, model,
            transform=lambda x, val: transforms.noise(x, intensity=val),
            name=name, transform_name="noise",
            min_val=0.0, max_val=0.1, samples=30)
    
    sweep(encoded_images, targets, model,
        transform=lambda x, val: transforms.crop(x, p=val),
        name=name, transform_name="crop",
        min_val=0.05, max_val=1.0, samples=50)

    model.module.set_distribution(transforms.training, n=DIST_SIZE)
    model.train()


def evaluate(model, image, target, test_transforms=False):
    
    if not isinstance(model, DecodingGramNet):
        model = nn.DataParallel(DecodingGramNet.load(distribution=transforms.identity,
                                            n=1, weights_file=model))

    image = im.torch(im.load(image)).unsqueeze(0)
    target = binary.parse(str(target))
    prediction = model(image).mean(dim=1).squeeze().cpu().data.numpy()
    prediction = binary.get(prediction)

    print (f"Target: {binary.str(target)}, Prediction: {binary.str(prediction)}, \
            Distance: {binary.distance(target, prediction)}")

    if test_transforms:
        sweep(image, [target], model,
                transform=lambda x, val: transforms.rotate(x, rand_val=False, theta=val),
                name="eval", transform_name="rotate",
                min_val=-0.6, max_val=0.6, samples=60)

def save_amnesia_data(model=None):
    if isinstance(model, str):
        model = nn.DataParallel(DecodingGramNet.load(distribution=transforms.encoding,
                                            n=48, weights_file=model))

    files = glob.glob(f"data/amnesia/*.pth")
    for batch in tqdm.tqdm(files[:100], ncols=50):
        perturbation, images, targets = torch.load(batch)
        perturbation.requires_grad = True
        encoded_ims, perturbation = encode_binary(images, targets, \
            model, max_iter=20, perturbation=perturbation, use_weighting=True)

        torch.save((perturbation.data, images.data, targets), f"{OUTPUT_DIR}{batch.split('/')[-1]}")

        

def encode_images(num_batches, batch_size, model, files):
    model = nn.DataParallel(DecodingGramNet.load(distribution=transforms.identity,
                                            n=ENCODING_DIST_SIZE, weights_file=model))
    model.eval()
    for i in range(num_batches):

        images = [im.load(image) for image in image_files]
        images = im.stack(images)
        targets = [binary.random(n=TARGET_SIZE) for _ in range(0, len(images))]
        model.eval()

if __name__ == "__main__":
    # save_amnesia_data(model="jobs/experiment57/output/train_test.pth")
    Fire()


