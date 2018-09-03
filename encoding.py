
import random, sys, os, json, glob, argparse

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from fire import Fire

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from models import DecodingModel, DataParallelModel
from torchvision import models
from logger import Logger, VisdomLogger
from utils import *

import IPython

import transforms


# LOGGING
logger = VisdomLogger("encoding", server="35.230.67.129", port=8000, env=JOB)
logger.add_hook(lambda x: logger.step(), feature="epoch", freq=20)
logger.add_hook(lambda x: logger.plot(x, "Encoding Loss", opts=dict(ymin=0)), feature="loss", freq=50)


""" 
Computes the changed images, given a a specified perturbation, standard deviation weighting, 
and epsilon.
"""


def compute_changed_images(images, perturbation, std_weights, epsilon=EPSILON):

    perturbation_w2 = perturbation * std_weights
    perturbation_zc = (
        perturbation_w2
        / perturbation_w2.view(perturbation_w2.shape[0], -1)
        .norm(2, dim=1, keepdim=True)
        .unsqueeze(2)
        .unsqueeze(2)
        .expand_as(perturbation_w2)
        * epsilon
        * (perturbation_w2[0].nelement() ** 0.5)
    )

    changed_images = (images + perturbation_zc).clamp(min=0.0, max=1.0)
    return changed_images


""" 
Computes the cross entropy loss of a set of encoded images, given the model and targets.
"""


def loss_func(model, x, targets):
    scores = model.forward(x)
    predictions = scores.mean(dim=1)
    score_targets = binary.target(targets).unsqueeze(1).expand_as(scores)

    return (F.binary_cross_entropy(scores, score_targets), predictions.cpu().data.numpy().round(2))


""" 
Encodes a set of images with the specified binary targets, for a given number of iterations.
"""


def encode_binary(
    images, targets, model=DecodingModel(), n=None, max_iter=500, verbose=False, perturbation=None, use_weighting=False
):

    if n is not None:
        if verbose:
            print(f"Changing distribution size: {model.n} -> {n}")
        n, model.n = (model.n, n)

    returnPerturbation = True
    if perturbation is None:
        perturbation = nn.Parameter(0.03 * torch.randn(images.size()).to(DEVICE) + 0.0)
        returnPerturbation = False

    changed_images = images.detach()
    optimizer = torch.optim.Adam([perturbation], lr=ENCODING_LR)
    std_weights = get_std_weight(images, alpha=PERT_ALPHA) if use_weighting else 1

    for i in range(0, max_iter):

        changed_images = compute_changed_images(images, perturbation, std_weights)
        loss, predictions = loss_func(model, changed_images, targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        error = np.mean([binary.distance(x, y) for x, y in zip(predictions, targets)])

        if verbose:
            logger.update("epoch", i)
            logger.update("loss", loss)
            logger.update("bits", error)

    changed_images = compute_changed_images(images, perturbation, std_weights)

    if n is not None:
        if verbose:
            print(f"Fixing distribution size: {model.n} -> {n}")
        n, model.n = (model.n, n)

    if returnPerturbation:
        return changed_images.detach(), perturbation.detach()

    return changed_images.detach()


""" 
Command-line interface for encoding a single image.
"""


def encode(
    image,
    out,
    target=binary.str(binary.random(TARGET_SIZE)),
    n=96,
    model=None,
    max_iter=500,
    use_weighting=True,
    perturbation_out=None,
):

    if not isinstance(model, DecodingModel):
        model = DataParallelModel(DecodingModel.load(distribution=transforms.encoding, n=n, weights_file=model))
    image = im.torch(im.load(image)).unsqueeze(0)
    print("Target: ", target)
    target = binary.parse(str(target))
    encoded = encode_binary(image, [target], model, n=n, verbose=True, max_iter=max_iter, use_weighting=use_weighting)
    im.save(im.numpy(encoded.squeeze()), file=out)
    if perturbation_out != None:
        im.save(im.numpy((image - encoded).squeeze()), file=perturbation_out)


if __name__ == "__main__":
    Fire(encode)
