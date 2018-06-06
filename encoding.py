
from __future__ import print_function

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import random, sys, os, json, glob, argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from models import DecodingNet
from torchvision import models
from logger import Logger
from utils import *

import IPython

import transforms

EPSILON = 9e-3
MIN_LOSS = 2e-3
BATCH_SIZE = 96


def encode_binary(images, targets, model=DecodingNet(), max_iter=200, verbose=False, perturbation=None):

	logger = Logger("encoding", ("loss", "bits"), verbose=verbose, print_every=50, plot_every=50)

	def loss_func(model, x):
		scores = model.forward(x)
		predictions = scores.mean(dim=1)
		score_targets = binary.target(targets).unsqueeze(1).expand_as(scores)

		return F.binary_cross_entropy(scores, score_targets), \
			predictions.cpu().data.numpy().round(2)
	
	returnPerturbation = True
	if not isinstance(perturbation, torch.Tensor):
		perturbation = nn.Parameter(0.03*torch.randn(images.size()).to(DEVICE)+0.0)
		returnPerturbation = False

	opt = torch.optim.Adam([perturbation], lr=5e-1)
	changed_images = images.detach()

	for i in range(0, max_iter+1):

		perturbation_zc = perturbation/perturbation.view(perturbation.shape[0], -1)\
			.norm(2, dim=1, keepdim=True).unsqueeze(2).unsqueeze(2).expand_as(perturbation)\
			*EPSILON*(perturbation[0].nelement()**0.5)

		changed_images = (images + perturbation_zc).clamp(min=0.0, max=1.0)

		loss, predictions = loss_func(model, changed_images)
		loss.backward()
		opt.step(); opt.zero_grad()

		error = np.mean([binary.distance(x, y) for x, y in zip(predictions, targets)])
		logger.step('loss', loss)
		logger.step('bits', error)

	if returnPerturbation:
		return changed_images.detach(), perturbation.detach()

	return changed_images.detach()

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description='Encodes a given image and prints corrupt bits.')
	parser.add_argument('--path', default=None, help="Path of model to load")
	args = parser.parse_args()

	def p(x):
		x = transforms.resize_rect(x)
		x = transforms.rotate(transforms.scale(x, 0.6, 1.4), max_angle=30)
		x = transforms.gauss(x, min_sigma=0.8, max_sigma=1.2)
		x = transforms.translate(x)
		x = transforms.identity(x)
		return x

	model = nn.DataParallel(DecodingNet(n=80, distribution=p))
	if args.path is not None: model.module.load(args.path)
	model.eval()

	images = [im.load(image) for image in ["images/house.png"]]
	images = im.stack(images)
	targets = [binary.random(n=TARGET_SIZE) for _ in range(0, len(images))]

	encode_binary(images, targets, model, verbose=True, max_iter=500)


