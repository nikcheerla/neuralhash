
import random, sys, os, json, glob, argparse

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from fire import Fire

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from models import DecodingNet, UNet
from torchvision import models
from logger import Logger
from utils import *

import IPython

import transforms

""" 
Encodes a set of images with the specified binary targets, for a given number of iterations.
"""
def encode_binary(images, targets, model=DecodingNet(), n=None,
					max_iter=200, verbose=False, perturbation=None, sfl=False):

	logger = Logger("encoding", ("loss", "bits", "unet_loss", "total"), verbose=verbose, print_every=5, plot_every=40)
	
	if n is not None: 
		if verbose: print (f"Changing distribution size: {model.module.n} -> {n}")
		n, model.module.n = (model.module.n, n)

	def loss_func(model, x):
		scores = model.forward(x)
		predictions = scores.mean(dim=1)
		score_targets = binary.target(targets).unsqueeze(1).expand_as(scores)

		return F.binary_cross_entropy(scores, score_targets), \
			predictions.cpu().data.numpy().round(2)
	
	def unet_loss_func(unet, changed_images, perturbation_zc):
		preds = unet(changed_images)
		return -1 * (preds - perturbation_zc).pow(2).sum() / changed_images.size(0)

	def viz_preds(model, x, y, name):
		preds = model(x)
		for i, (pred, truth, enc) in enumerate(zip(preds, y, x)):
			im.save(im.numpy(enc), f'{OUTPUT_DIR}{i}_encoded_{name}.jpg')
			im.save(3*np.abs(im.numpy(pred)), f'{OUTPUT_DIR}{i}_pred_{name}.jpg')
			im.save(3*np.abs(im.numpy(truth)), f'{OUTPUT_DIR}{i}_truth_{name}.jpg')

	returnPerturbation = True
	if not isinstance(perturbation, torch.Tensor):
		perturbation = nn.Parameter(0.03*torch.randn(images.size()).to(DEVICE)+0.0)
		returnPerturbation = False

	opt = torch.optim.Adam([perturbation], lr=1e-3)
	changed_images = images.detach()

	####### LOAD IN UNet #########
	unet = nn.DataParallel(UNet())
	unet.module.load('jobs/experiment_unet/output/train_unet.pth')
	unet.eval()

	for i in range(0, max_iter+1):

		perturbation_zc = perturbation/perturbation.view(perturbation.shape[0], -1)\
			.norm(2, dim=1, keepdim=True).unsqueeze(2).unsqueeze(2).expand_as(perturbation)\
			*EPSILON*(perturbation[0].nelement()**0.5)

		changed_images = (images + perturbation_zc).clamp(min=0.0, max=1.0)

		unet_loss = unet_loss_func(unet, changed_images, perturbation_zc)
		loss, predictions = loss_func(model, changed_images)

		if sfl:
			total = loss + 0.01*unet_loss
		else:
			total = loss

		total.backward()
		opt.step(); opt.zero_grad()

		error = np.mean([binary.distance(x, y) for x, y in zip(predictions, targets)])
		logger.step('loss', loss)
		logger.step('bits', error)
		logger.step('unet_loss', unet_loss)
		logger.step('total', total)

	if sfl:
		name = "sfl" 
	else:
		name = "base"
	viz_preds(unet, changed_images, perturbation_zc, name)

	if n is not None: 
		if verbose: print (f"Fixing distribution size: {model.module.n} -> {n}")
		n, model.module.n = (model.module.n, n)

	if returnPerturbation:
		return changed_images.detach(), perturbation.detach()

	return changed_images.detach()


""" 
Command-line interface for encoding a single image
"""
def encode(image, out, target=binary.str(binary.random(TARGET_SIZE)), n=96,
			model=None, max_iter=300):

	if model is None:
		model = nn.DataParallel(DecodingNet(distribution=transforms.easy, n=n))

	if type(model) is str:
		x = nn.DataParallel(DecodingNet(distribution=transforms.easy, n=n))
		x.module.load(model)
		model = x

	image = im.torch(im.load(image)).unsqueeze(0)
	print ("Target: ", target)
	target = binary.parse(str(target))
	encoded = encode_binary(image, [target], model, n=n, verbose=True, max_iter=max_iter)
	im.save(im.numpy(encoded.squeeze()), file=out)


if __name__ == "__main__":
	Fire(encode)
