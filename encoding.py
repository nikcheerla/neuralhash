
from __future__ import print_function

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import random, sys, os, json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from models import DecodingNet
from torchvision import models
from utils import *

import IPython

import transforms

EPSILON = 1.5e-2
MIN_LOSS = 5e-3
BATCH_SIZE = 8


def encode_binary(image, model, target, max_iter=200, verbose=False):

	if not isinstance(image, torch.Tensor):
		image = im.torch(image)
	perturbation = nn.Parameter(0.03*torch.randn(image.size()).to(DEVICE)+0.0)

	# returns an image after a series of transformations
	def p(x):
		x = transforms.resize_rect(x)
		x = transforms.rotate(transforms.scale(x, 0.6, 1.4), max_angle=30)
		x = transforms.gauss(x, min_sigma=0.8, max_sigma=1.2)
		x = transforms.translate(x)
		x = transforms.identity(x)
		return x

	# returns the loss for the image
	def loss_func(model, x):
		scores = model.forward(x, distribution=p, n=BATCH_SIZE, return_variance=False) # (N, T)
		predictions = scores.mean(dim=0)
		#smoothness_loss = tv_loss(x, 2e-2) #second parameter is tv_loss
		return F.binary_cross_entropy(scores, binary.target(target).repeat(BATCH_SIZE, 1)), \
			predictions.cpu().data.numpy().round(2)
		# return F.mse_loss(predictions, binary.target(target).repeat(BATCH_SIZE, 1)), \
		#             predictions.mean(dim=0).cpu().data.numpy().round(2)
	
	alpha, beta, gamma = 1, 1e-6, 0.01
	hinge = 0.3

	opt = torch.optim.Adam([perturbation], lr=2e-1)
	losses, preds = [], []

	for i in range(0, max_iter+1):
		#print('shape: ' + str(image.size()))
		opt.zero_grad()
		#TODO: Change back
		#perturbation_zc = (perturbation - perturbation.mean())/perturbation.std()*EPSILON
		perturbation_zc = (perturbation)/(perturbation.norm(2))*EPSILON
		changed_image = (image + perturbation_zc).clamp(min=0, max=1)

		loss, predictions = loss_func(model, changed_image)
		loss += tve_loss(perturbation, 2e-3)
		#loss += tv_loss(perturbation
		# perceptual_loss = beta*tve_loss(changed_image) + gamma*perturbation.norm(2)
		# robustness_loss = alpha*((mean_loss-hinge).clamp(min=0))
		# loss = perceptual_loss + robustness_loss
		loss.backward()
		opt.step()

		preds.append(predictions)
		losses.append(loss.cpu().data.numpy())

		if verbose and i % 20 == 0:
			# print("Epsilon: ", eps.cpu().data.numpy())
			# print (perceptual_loss.cpu().data.numpy(), robustness_loss.cpu().data.numpy())
			print ("Loss: ", np.mean(losses[-20:]))

			im.save(im.numpy(perturbation), file=f"{OUTPUT_DIR}perturbation.jpg")
			im.save(im.numpy(changed_image), file=f"{OUTPUT_DIR}changed_image.jpg")

			plt.plot(np.array(preds)); 
			plt.savefig(OUTPUT_DIR + "preds.jpg"); plt.cla()
			plt.plot(losses); 
			plt.savefig(OUTPUT_DIR + "loss.jpg"); plt.cla()

			pred = binary.get(np.mean(preds[-20:], axis=0))
			print ("Modified prediction: ", binary.str(pred), binary.distance(pred, target))

		smooth_loss = np.mean(losses[-20:])
		if smooth_loss <= MIN_LOSS:
			break

	print(f"Epsilon: {EPSILON}")
	print ("Loss: ", np.mean(losses[-1]))
	return im.numpy(changed_image)

if __name__ == "__main__":
	target = binary.random(n=TARGET_SIZE)
	model = DecodingNet()
	print("Target: ", binary.str(target))
	encode_binary(im.load("images/car.jpg"), model, target=target, verbose=True)


