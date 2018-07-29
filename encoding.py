
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

from scipy.ndimage.filters import uniform_filter
from scipy.signal import convolve2d
import IPython

import transforms

""" 
Encodes a set of images with the specified binary targets, for a given number of iterations.
"""
def encode_binary(images, targets, model=DecodingNet(), n=None,
					max_iter=200, verbose=False, perturbation=None, sfl=False, perceptual=False, alpha=0.8):

	logger = Logger("encoding", ("loss", "bits", "unet_loss", "total", "content_loss"), verbose=verbose, print_every=5, plot_every=40)
	
	if n is not None: 
		if verbose: print (f"Changing distribution size: {model.module.n} -> {n}")
		n, model.module.n = (model.module.n, n)

	def loss_func(model, x):
		scores = model.forward(x)
		predictions = scores.mean(dim=1)
		score_targets = binary.target(targets).unsqueeze(1).expand_as(scores)
		return F.binary_cross_entropy(scores, score_targets), \
			predictions.cpu().data.numpy().round(2)
	
	def get_std_weight(images, n=5, alpha=0.5):
		N, C, H, W = images.shape
		kernel = kernel = torch.ones(C, 1, n, n)
		with torch.no_grad():
			padded = F.pad(images, (n//2, n//2, n//2, n//2), mode='replicate')
			sums = F.conv2d(padded, kernel.to(DEVICE), groups=3, padding=0) * (1/(n**2))
			sums_2 = F.conv2d(padded**2, kernel.to(DEVICE), groups=3, padding=0) * (1/(n**2))
			stds = (sums_2 - (sums**2) + 1e-5)**alpha
		return stds


	def unet_loss_func(unet, changed_images, perturbation_zc, original_images):
		preds, f_c = unet(changed_images)
		preds_orig, f_o = unet(original_images)
		content_loss = F.mse_loss(f_c[0], f_o[0]) + F.mse_loss(f_c[1], f_o[1]) + \
			F.mse_loss(f_c[2], f_o[2]) + F.mse_loss(f_c[3], f_o[3] + F.mse_loss(f_c[4], f_c[4]))
		preds_loss = ((preds_orig - preds)**2).view(preds.size(0), -1).sum(1).mean()
		return preds_loss, content_loss#(preds - perturbation_zc).pow(2).sum() / changed_images.size(0)

	def viz_preds(model, x, y, name):
		preds, _ = model(x)
		for i, (pred, truth, enc) in enumerate(zip(preds, y, x)):
			im.save(im.numpy(enc), f'{OUTPUT_DIR}{i}_encoded_{name}.jpg')
			im.save(4*np.abs(im.numpy(pred)), f'{OUTPUT_DIR}{i}_pred_{name}.jpg')
			im.save(4*np.abs(im.numpy(truth)), f'{OUTPUT_DIR}{i}_truth_{name}.jpg')

	returnPerturbation = True
	if not isinstance(perturbation, torch.Tensor):
		perturbation = nn.Parameter(0.03*torch.randn(images.size()).to(DEVICE)+0.0)
		returnPerturbation = False

	opt = torch.optim.Adam([perturbation], lr=3e-3)
	changed_images = images.detach()
	if perceptual:
		std_weights = get_std_weight(images, alpha=alpha).detach()
		for i, weight in enumerate(std_weights):
			im.save(im.numpy(weight), f'{OUTPUT_DIR}{i}_std_weights.jpg')

	####### LOAD IN UNet #########
	unet = nn.DataParallel(UNet())
	unet.module.load('jobs/experiment_unet/output/train_unet.pth')
	unet.eval()
	unet_weight = 0.1

	for i in range(0, max_iter+1):

		if perceptual:
			w_pert = perturbation * std_weights
			perturbation_zc = w_pert/w_pert.view(w_pert.shape[0], -1)\
				.norm(2, dim=1, keepdim=True).unsqueeze(2).unsqueeze(2).expand_as(w_pert)\
				*EPSILON*(w_pert[0].nelement()**0.5)
		else:
			perturbation_zc = perturbation/perturbation.view(perturbation.shape[0], -1)\
				.norm(2, dim=1, keepdim=True).unsqueeze(2).unsqueeze(2).expand_as(perturbation)\
				*EPSILON*(perturbation[0].nelement()**0.5)

		changed_images = (images + perturbation_zc).clamp(min=0.0, max=1.0)

		unet_loss, content_loss = unet_loss_func(unet, changed_images, perturbation_zc, images)
		loss, predictions = loss_func(model, changed_images)

		if sfl:
			total = loss + unet_weight*unet_loss
		else:
			total = loss.mean()

		total.backward()
		opt.step(); opt.zero_grad()

		error = np.mean([binary.distance(x, y) for x, y in zip(predictions, targets)])
		logger.step('loss', loss)
		logger.step('bits', error)
		logger.step('unet_loss', unet_loss)
		logger.step('content_loss', content_loss)
		logger.step('total', total)

	if perceptual:
		name = "perc"
	if sfl:
		name = "sfl" 
	if not sfl and not perceptual:
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
