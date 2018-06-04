
import numpy as np
import random, sys, os, json, glob

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import *
import transforms
from encoding import encode_binary
from models import DecodingNet

from skimage.morphology import binary_dilation
import IPython

from testing import test_transforms

def p(x):
    x = transforms.resize_rect(x)
    x = transforms.rotate(transforms.scale(x, 0.6, 1.4), max_angle=30)
    x = transforms.gauss(x, min_sigma=0.8, max_sigma=1.2)
    x = transforms.translate(x)
    x = transforms.identity(x)
    return x

def simple(x):
    x = transforms.rotate(transforms.scale(x, 0.6, 1.4), max_angle=30)
    return x

def loss_func(model, transformed_ims, targets):
	targets = torch.cat([binary.target(target).unsqueeze(0) for target in targets], dim=0)
	predictions = model.forward_batch(transformed_ims) # (N, T)
	return F.binary_cross_entropy(predictions, torch.tensor(targets))
	# predictions = model.forward(transformed_ims[0], distribution=p, n=64)
	# return F.binary_cross_entropy(predictions, binary.target(targets[0]).repeat(64, 1))

def loss_func2(model, encoded_im, target):
	predictions = model.forward(p(encoded_im), distribution=p, n=64) # (N, T)
	return F.binary_cross_entropy(predictions, binary.target(target).repeat(64, 1))

def encode_images(model, batch, targets):
	encoded_ims = []
	for i, img in enumerate(batch):
		encoded_ims.append(im.torch(encode_binary(img, model, targets[i], max_iter=10, verbose=False)))
	return encoded_ims

def create_targets():
	targets = []
	for i in range(TARGET_SIZE):
		target = [0 for j in range(TARGET_SIZE)]
		target[i] = 1
		targets.append(target)
	return targets 

if __name__ == "__main__":

	BATCH_SIZE = 10

	model = DecodingNet()
	optimizer = torch.optim.Adam(model.features.classifier.parameters(), lr=1e-4)
	
	def data_generator():
		# path = "/home/RC/neuralhash/data/tiny-imagenet-200/test/images"
		path = "data/cats/n02497673_7861.jpg"
		# files = glob.glob(f"{path}/*.JPEG")
		files = glob.glob(path)
		for image in random.sample(files, k=len(files)):
			img = im.load(image)
			if img is None:
				continue
			yield img

	losses = []
	targets = [binary.random(n=TARGET_SIZE) for x in range(200)]
	for i in range(0, 1):
		for target in targets:
			image = next(data_generator())
			# ind = random.randint(0,len(targets)-1)
			# target = binary.random(n=TARGET_SIZE)#targets[ind]
			encoded_im = im.torch(encode_binary(image, model, target, max_iter=5, verbose=False))
			loss = loss_func2(model, encoded_im, target)
			# batch = next(batched(data_generator(), batch_size=BATCH_SIZE))
			# targets = [binary.random(n=TARGET_SIZE) for x in range(BATCH_SIZE)]
			# encoded_ims = encode_images(model, batch, targets)
			# transformed_ims = torch.cat([p(x).unsqueeze(0) for x in encoded_ims], dim=0)
			# loss = loss_func(model, transformed_ims, targets)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			losses.append(loss.cpu().data.numpy())

			if i % 10 == 0:
				model.drawLastLayer(OUTPUT_DIR + "mat_viz_" + str(i) + ".png")
			print("train loss = ", np.mean(losses[-1]))
			# print("loss after step = ", loss_func(model, encoded_im, target).cpu().data.numpy())

	test_transforms(model)
	model.save(OUTPUT_DIR + "train_test.pth")


