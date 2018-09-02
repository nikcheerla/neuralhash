
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

from models import DecodingNet, DecodingGramNet
from modules import UNet
from torchvision import models
from logger import Logger
from utils import *

import IPython

import transforms

""" 
Encodes a set of images with the specified binary targets, for a given number of iterations.
"""
def encode_binary(images, targets, model=DecodingNet(), n=None,
					max_iter=200, verbose=False, perturbation=None, use_weighting=False, sfl=False):

	logger = Logger("encoding", ("loss", "bits", "unet_loss", "total"), verbose=verbose, print_every=5, plot_every=40)
	
	if n is not None: 
		if verbose: print (f"Changing distribution size: {model.module.n} -> {n}")
		n, model.module.n = (model.module.n, n)

	def loss_func(model, x):
		scores = model.forward(x)
		predictions = scores.mean(dim=1)
		score_targets = binary.target(targets).unsqueeze(1).expand_as(scores)

		return F.binary_cross_entropy(scores, score_targets), \
			scores.cpu().data.numpy().round(2)

	def norm_bound(pert):
		return pert/pert.view(pert.shape[0], -1)\
			.norm(2, dim=1, keepdim=True).unsqueeze(2).unsqueeze(2).expand_as(pert)\
			*EPSILON*(pert[0].nelement()**0.5)

	def unet_loss_func(unet, changed_images, perturbation_zc, original_images):
		preds = unet(changed_images)
		preds_orig = unet(original_images)
		preds_loss = ((preds_orig - preds)**2).view(preds.size(0), -1).sum(1).mean()
		return preds_loss

	####### LOAD IN UNet #########
	if sfl:
		unet = nn.DataParallel(UNet())
		unet.module.load('jobs/experiment60/output/train_unet.pth')
		unet.eval()
	unet_loss = 0
	unet_weight = 0.2


	returnPerturbation = True
	if not isinstance(perturbation, torch.Tensor):
		perturbation = nn.Parameter(0.03*torch.randn(images.size()).to(DEVICE)+0.0)
		optimizer = torch.optim.Adam([perturbation], lr=ENCODING_LR)
		returnPerturbation = False

	optimizer = torch.optim.Adam([perturbation], lr=ENCODING_LR)

	if use_weighting:
		std_weights = get_std_weight(images, alpha=PERT_ALPHA).detach()
	
	changed_images = images.detach()

	for i in range(0, max_iter):

		w_pert = perturbation*std_weights if use_weighting else perturbation
		perturbation_zc = norm_bound(w_pert)

		changed_images = (images + perturbation_zc).clamp(min=0.0, max=1.0)

		if sfl:
			unet_loss = unet_loss_func(unet, changed_images, perturbation_zc, images)
		loss, preds = loss_func(model, changed_images)

		if sfl:
			total = loss + unet_weight * unet_loss
		else:
			total = loss


		total.backward()
		optimizer.step(); optimizer.zero_grad()

		error = np.mean((np.floor(preds*2) - np.repeat(np.expand_dims(targets, 1), preds.shape[1], axis=1))**2, axis=(0,1)).sum()
		# error = np.mean([binary.distance(x, y) for x, y in zip(predictions, targets)])
		logger.step('loss', loss)
		logger.step('bits', error)
		logger.step('unet_loss', unet_loss)
		logger.step('total', total)

	if n is not None: 
		if verbose: print (f"Fixing distribution size: {model.module.n} -> {n}")
		n, model.module.n = (model.module.n, n)

	w_pert = perturbation*std_weights if use_weighting else perturbation
	perturbation_zc = norm_bound(w_pert)

	changed_images = (images + perturbation_zc).clamp(min=0.0, max=1.0)

	if returnPerturbation:
		return changed_images.detach(), perturbation.detach()

	return changed_images.detach().data


""" 
Command-line interface for encoding a single image
"""
def encode(image, out, target=binary.str(binary.random(TARGET_SIZE)), n=96,
			model=None, max_iter=300):
	if not isinstance(model, DecodingGramNet):
		model = nn.DataParallel(DecodingGramNet.load(distribution=transforms.encoding,
											n=n, weights_file=model))
	image = im.torch(im.load(image)).unsqueeze(0)
	print ("Target: ", target)
	target = binary.parse(str(target))
	encoded = encode_binary(image, [target], model, n=n, verbose=True, max_iter=max_iter, use_weighting=True)
	im.save(im.numpy(encoded.squeeze()), file=out)


if __name__ == "__main__":
	Fire(encode)
