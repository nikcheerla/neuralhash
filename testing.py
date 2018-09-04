
import random, sys, os, glob, pickle
import argparse, tqdm

import matplotlib as mpl

mpl.use("Agg")
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
from models import BaseModel, DecodingModel, DataParallelModel
from logger import Logger, VisdomLogger

import IPython


# LOGGING
logger = VisdomLogger("test", server="35.230.67.129", port=8000, env=JOB)
logger.add_hook(lambda x: logger.step(), feature="orig", freq=1)


def sweep(images, targets, model, transform, name, samples=10):

    min_val, max_val = transform.plot_range

    results = []
    for val in tqdm.tqdm(np.linspace(min_val, max_val, samples), ncols=30):
        transformed = transform(images, val)
        predictions = model(transformed).mean(dim=1).cpu().data.numpy()

        mse_loss = np.mean([binary.mse_dist(x, y) for x, y in zip(predictions, targets)])
        binary_loss = np.mean([binary.distance(x, y) for x, y in zip(predictions, targets)])
        results.append((val, binary_loss, mse_loss))

    x, bits_off, mse = (np.array(x) for x in zip(*results))

    print(transform.__name__, np.mean(bits_off))
    logger.update(transform.__name__, np.mean(bits_off))

    np.savez_compressed(f"output/{name}_{transform.__name__}.npz", x=x, bits_off=bits_off, mse=mse)
    # logger.viz(f"{name}_{transform_name}", method='line',
    #         Y=np.column_stack((32*mse, bits_off)),
    #         X=np.column_stack((x, x)),
    #         opts=dict(title=f"{name}_{transform_name}", ylim=(0, 16),
    #             legend=['squared error', 'bits'])
    # )

    fig, ax1 = plt.subplots()
    ax1.plot(x, bits_off, "b")
    ax1.set_ylim(0, TARGET_SIZE // 2)
    ax1.set_ylabel("Number Incorrect Bits")
    ax2 = ax1.twinx()
    ax2.plot(x, mse, "r")
    ax2.set_ylim(0, 0.25)
    ax2.set_ylabel("Mean Squared Error")
    plt.savefig(f"output/{name}_{transform.__name__}.jpg")
    plt.cla()
    plt.clf()
    plt.close()
    return np.mean(bits_off)


def test_transforms(model=None, image_files=VAL_FILES, name="test", max_iter=250):

    if not isinstance(model, BaseModel):
        print(f"Loading model from {model}")
        model = DataParallelModel(
            DecodingModel.load(distribution=transforms.new_dist, n=ENCODING_DIST_SIZE, weights_file=model)
        )

    images = [im.load(image) for image in image_files]
    images = im.stack(images)
    targets = [binary.random(n=TARGET_SIZE) for _ in range(0, len(images))]
    model.eval()

    encoded_images = encode_binary(
        images, targets, model, n=ENCODING_DIST_SIZE, verbose=True, max_iter=max_iter, use_weighting=True
    )

    logger.images(images, "original_images", resize=196)
    logger.images(encoded_images, "encoded_images", resize=196)
    for img, encoded_im, filename, target in zip(images, encoded_images, image_files, targets):
        im.save(im.numpy(img), file=f"output/_{binary.str(target)}_original_{filename.split('/')[-1]}")
        im.save(im.numpy(encoded_im), file=f"output/_{binary.str(target)}_encoded_{filename.split('/')[-1]}")

    model.set_distribution(transforms.identity, n=1)
    predictions = model(encoded_images).mean(dim=1).cpu().data.numpy()
    binary_loss = np.mean([binary.distance(x, y) for x, y in zip(predictions, targets)])

    for transform in [
        transforms.pixilate,
        transforms.blur,
        transforms.rotate,
        transforms.scale,
        transforms.translate,
        transforms.noise,
        transforms.crop,
        transforms.gauss,
        transforms.whiteout,
        transforms.resize_rect,
        transforms.color_jitter,
        transforms.jpeg_transform,
        transforms.elastic,
        transforms.brightness,
        transforms.contrast,
        transforms.flip,
    ]:
        sweep(encoded_images, targets, model, transform=transform, name=name, samples=60)

    # sweep(
    #     encoded_images,
    #     targets,
    #     model,
    #     transform=lambda x, val: transforms.rotate(x, rand_val=False, theta=val),
    #     name=name,
    #     transform_name="rotate",
    #     min_val=-0.6,
    #     max_val=0.6,
    #     samples=80,
    # )

    # sweep(
    #     encoded_images,
    #     targets,
    #     model,
    #     transform=lambda x, val: transforms.scale(x, rand_val=False, scale_val=val),
    #     name=name,
    #     transform_name="scale",
    #     min_val=0.6,
    #     max_val=1.4,
    #     samples=50,
    # )

    # sweep(
    #     encoded_images,
    #     targets,
    #     model,
    #     transform=lambda x, val: transforms.translate(x, rand_val=False, radius=val),
    #     name=name,
    #     transform_name="translate",
    #     min_val=0.0,
    #     max_val=1.0,
    #     samples=50,
    # )

    # sweep(
    #     encoded_images,
    #     targets,
    #     model,
    #     transform=lambda x, val: transforms.noise(x, intensity=val),
    #     name=name,
    #     transform_name="noise",
    #     min_val=0.0,
    #     max_val=0.1,
    #     samples=30,
    # )

    # sweep(
    #     encoded_images,
    #     targets,
    #     model,
    #     transform=lambda x, val: transforms.crop(x, p=val),
    #     name=name,
    #     transform_name="crop",
    #     min_val=0.1,
    #     max_val=1.0,
    #     samples=50,
    # )

    # sweep(
    #     encoded_images,
    #     targets,
    #     model,
    #     transform=lambda x, val: transforms.gauss(x, sigma=val, rand_val=False),
    #     name=name,
    #     transform_name="gauss",
    #     min_val=0.3,
    #     max_val=4,
    #     samples=50,
    # )

    # sweep(
    #     encoded_images,
    #     targets,
    #     model,
    #     transform=lambda x, val: transforms.whiteout(x, scale=val, rand_val=False),
    #     name=name,
    #     transform_name="whiteout",
    #     min_val=0.02,
    #     max_val=0.2,
    #     samples=50,
    # )

    # sweep(
    #     encoded_images,
    #     targets,
    #     model,
    #     transform=lambda x, val: transforms.resize_rect(x, ratio=val, rand_val=False),
    #     name=name,
    #     transform_name="resize_rect",
    #     min_val=0.5,
    #     max_val=1.5,
    #     samples=50,
    # )

    # sweep(
    #     encoded_images,
    #     targets,
    #     model,
    #     transform=lambda x, val: transforms.color_jitter(x, jitter=val),
    #     name=name,
    #     transform_name="jitter",
    #     min_val=0,
    #     max_val=0.2,
    #     samples=50,
    # )

    # sweep(
    #     encoded_images,
    #     targets,
    #     model,
    #     transform=lambda x, val: transforms.convertToJpeg(x, q=val),
    #     name=name,
    #     transform_name="jpg",
    #     min_val=10,
    #     max_val=100,
    #     samples=50,
    # )

    logger.update("orig", binary_loss)
    model.set_distribution(transforms.training, n=DIST_SIZE)
    model.train()


def evaluate(model, image, target, test_transforms=False):

    if not isinstance(model, BaseModel):
        model = DataParallelModel(DecodingModel.load(distribution=transforms.identity, n=1, weights_file=model))

    image = im.torch(im.load(image)).unsqueeze(0)
    target = binary.parse(str(target))
    prediction = model(image).mean(dim=1).squeeze().cpu().data.numpy()
    prediction = binary.get(prediction)

    # print (f"Target: {binary.str(target)}, Prediction: {binary.str(prediction)}, \
    #         Distance: {binary.distance(target, prediction)}")

    if test_transforms:
        sweep(image, [target], model, transform=transforms.rotate, name="eval", samples=60)


def test_transfer(model=None, image_files=VAL_FILES, max_iter=250):
    if not isinstance(model, BaseModel):
        print(f"Loading model from {model}")
        model = DataParallelModel(
            DecodingModel.load(distribution=transforms.encoding, n=ENCODING_DIST_SIZE, weights_file=model)
        )

    images = [im.load(image) for image in image_files]
    images = im.stack(images)
    targets = [binary.random(n=TARGET_SIZE) for _ in range(0, len(images))]
    model.eval()

    transform_list = [
        transforms.rotate,
        transforms.translate,
        transforms.scale,
        transforms.resize_rect,
        transforms.crop,
        transforms.whiteout,
        transforms.elastic,
        transforms.motion_blur,
        transforms.brightness,
        transforms.contrast,
        transforms.pixilate,
        transforms.blur,
        transforms.color_jitter,
        transforms.gauss,
        transforms.noise,
        transforms.impulse_noise,
        transforms.flip,
    ]

    labels = [t.__name__ for t in transform_list]
    score_matrix = np.zeros((len(transform_list), len(transform_list)))

    for i, t1 in enumerate(transform_list):
            
        model.set_distribution(lambda x: t1.random(x), n=ENCODING_DIST_SIZE)
        encoded_images = encode_binary(
            images, targets, model, n=ENCODING_DIST_SIZE, verbose=True, max_iter=max_iter, use_weighting=True
        )

        model.set_distribution(transforms.identity, n=1)
        t1_error = sweep(encoded_images, targets, model, transform=t1, name=f"{t1.__name__}", samples=60)
        
        for j, t2 in enumerate(transform_list):
            if t1.__name__ == t2.__name__: 
                score_matrix[i,j] = t1_error
                continue
            t2_error = sweep(encoded_images, targets, model, transform=t2, name=f'{t1.__name__}->{t2.__name__}', samples=60)
            score_matrix[i,j] = t2_error

            print(f'{t1.__name__} --> {t2.__name__}: {t2_error}')
    np.save('labels', labels)
    np.save('score_matrix', score_matrix)
    create_heatmap(score_matrix, labels)


if __name__ == "__main__":
    Fire()
