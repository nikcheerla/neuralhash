
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
from logger import Logger

from skimage.morphology import binary_dilation
import IPython

from testing import test_transforms



logger = Logger("train", ("bce", "bits"))

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
	logger = 
	return F.binary_cross_entropy(scores, binary.target(target).expand(BATCH_SIZE, TARGET_SIZE)), \
		predictions.cpu().data.numpy().round(2)
	# return F.mse_loss(predictions, binary.target(target).repeat(BATCH_SIZE, 1)), \
	#             predictions.mean(dim=0).cpu().data.numpy().round(2)

def loss_func(model, encoded_im, target):
	
	scores = model.forward(encoded_im, distribution=p, n=64) # (N, T)
	predictions = scores.mean(dim=0)
	bce_loss = F.binary_cross_entropy(scores, \
		binary.target(target).repeat(64, 1))

	logger.step("bce", bce_loss)

	return bce_loss, predictions.cpu().data.numpy().round(2)

def create_targets():
	return [(i == j) for i, j in product(range(TARGET_SIZE), range(TARGET_SIZE))]

if __name__ == "__main__":

	model = DecodingNet()
	model.train()
	optimizer = torch.optim.Adadelta(model.classifier.parameters(), lr=1.5e-2)
	
	def data_generator():

		files = glob.glob("data/colornet/*.jpg")
		while True:
			img = im.load(random.choice(files))
			if img is None: continue
			yield img, target	

	def checkpoint():
		print (f"Saving model to {OUTPUT_DIR}train_test.pth")
		model.save(OUTPUT_DIR + "train_test.pth")

	logger.add_hook(checkpoint)

	for images, targets in batched(data_generator()):
		images = im.stack(images)
		encoded_images = encode_binary(images, model, target, verbose=False, max_iter=4)
		loss, predictions = loss_func(model, encoded_images, target)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step() 

		logger.step ("bits", binary.distance(predictions, target))

		if (i+1) % 400 == 0:
			test_transforms(model)
	


