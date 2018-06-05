
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



logger = Logger("train", ("loss", "bits"), print_every=4, plot_every=40)

def loss_func(model, x, targets):
	scores = model.forward(x)
	predictions = scores.mean(dim=1)
	score_targets = binary.target(targets).unsqueeze(1).expand_as(scores)

	return F.binary_cross_entropy(scores, score_targets), \
		predictions.cpu().data.numpy().round(2)

def create_targets():
	return [(i == j) for i, j in product(range(TARGET_SIZE), range(TARGET_SIZE))]

if __name__ == "__main__":

	def p(x):
	    x = transforms.resize_rect(x)
	    x = transforms.rotate(transforms.scale(x, 0.6, 1.4), max_angle=30)
	    x = transforms.gauss(x, min_sigma=0.8, max_sigma=1.2)
	    x = transforms.translate(x)
	    x = transforms.identity(x)
	    return x

	model = nn.DataParallel(DecodingNet(n=48, distribution=p))
	optimizer = torch.optim.Adadelta(model.module.classifier.parameters(), lr= (1.5e-2)*48)
	model.train()
		
	def data_generator():

		files = glob.glob("data/colornet/*.jpg")
		while True:
			img = im.load(random.choice(files))
			if img is None: continue
			target = binary.random(TARGET_SIZE)
			yield img, target	

	def checkpoint():
		print (f"Saving model to {OUTPUT_DIR}train_test.pth")
		model.module.save(OUTPUT_DIR + "train_test.pth")

	logger.add_hook(checkpoint)

	for i, (images, targets) in enumerate(batched(data_generator(), batch_size=48)):
		images = im.stack(images)

		encoded_images = encode_binary(images, targets, model, verbose=False, max_iter=3)
		loss, predictions = loss_func(model, encoded_images, targets)
		logger.step ("loss", loss)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step() 

		error = np.mean([binary.distance(x, y) for x, y in zip(predictions, targets)])
		logger.step ("bits", error)

		if i == 1000: break
		if (i+1) % 100 == 0:
			test_transforms(model)

		
	


