
import numpy as np
import random, sys, os, json, glob
import tqdm, itertools, shutil

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

import torch

torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import *
import transforms
from encoding import encode_binary
from models import DecodingModel, DataParallelModel
from logger import Logger, VisdomLogger

from skimage.morphology import binary_dilation
import IPython

from testing import test_transforms


def loss_func(model, x, targets):
    scores = model.forward(x)
    predictions = scores.mean(dim=1)
    score_targets = binary.target(targets).unsqueeze(1).expand_as(scores)

    return (F.binary_cross_entropy(scores, score_targets), predictions.cpu().data.numpy().round(2))


def init_data(output_path, n=None):

    shutil.rmtree(output_path)
    os.makedirs(output_path)

    image_files = TRAIN_FILES
    if n is not None:
        image_files = image_files[0:n]

    for k, files in tqdm.tqdm(list(enumerate(batch(image_files, batch_size=BATCH_SIZE))), ncols=50):

        images = im.stack([im.load(img_file) for img_file in files]).detach()
        perturbation = nn.Parameter(0.03 * torch.randn(images.size()).to(DEVICE) + 0.0)
        targets = [binary.random(n=TARGET_SIZE) for i in range(len(images))]
        torch.save((perturbation.data, images.data, targets), f"{output_path}/{k}.pth")


if __name__ == "__main__":

    model = DataParallelModel(DecodingModel(n=DIST_SIZE, distribution=transforms.training))
    params = itertools.chain(model.module.classifier.parameters(), model.module.features[-1].parameters())
    optimizer = torch.optim.Adam(params, lr=2.5e-3)
    init_data("data/amnesia")

    logger = VisdomLogger("train", server="35.230.67.129", port=8000, env=JOB)
    logger.add_hook(lambda x: logger.step(), feature="epoch", freq=20)
    logger.add_hook(lambda data: logger.plot(data, "train_loss"), feature="loss", freq=50)
    logger.add_hook(lambda data: logger.plot(data, "train_bits"), feature="bits", freq=50)
    logger.add_hook(lambda x: model.save("output/train_test.pth", verbose=True), feature="epoch", freq=100)
    model.save("output/train_test.pth", verbose=True)

    files = glob.glob(f"data/amnesia/*.pth")
    for i, save_file in enumerate(random.choice(files) for i in range(0, 2701)):

        perturbation, images, targets = torch.load(save_file)
        perturbation = perturbation.requires_grad_()

        perturbation.requires_grad = True
        encoded_ims, perturbation = encode_binary(
            images, targets, model, max_iter=1, perturbation=perturbation, use_weighting=True
        )

        loss, predictions = loss_func(model, encoded_ims, targets)
        error = np.mean([binary.distance(x, y) for x, y in zip(predictions, targets)])

        logger.update("epoch", i)
        logger.update("loss", loss)
        logger.update("bits", error)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        torch.save((perturbation.data, images.data, targets), save_file)

        if i != 0 and i % 300 == 0:

            model.save("output/train_test.pth")
            model2 = DataParallelModel(
                DecodingModel.load(distribution=transforms.training, n=DIST_SIZE, weights_file="output/train_test.pth")
            )
            # test_transforms(model, random.sample(TRAIN_FILES, 16), name=f'iter{i}_train')
            test_transforms(model2, VAL_FILES, name=f"iter{i}_test", max_iter=300)
