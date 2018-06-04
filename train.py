
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
		# path = "/home/RC/neuralhash/data/tiny-imagenet-200/test/images"
		path = "data/colornet/*.jpg"
		files = glob.glob(path)
		while True:
			img = im.load(random.choice(files))
			if img is None: continue
			yield img	

	def checkpoint():
		print (f"Saving model to {OUTPUT_DIR}train_test.pth")
		model.save(OUTPUT_DIR + "train_test.pth")

	logger.add_hook(checkpoint)

	for i in range(0, 10000):

		image = im.torch(next(data_generator()))
		target = binary.random(n=TARGET_SIZE)

		encoded_im = im.torch(encode_binary(image, model, target, \
		 	verbose=False, max_iter=4))
		
		loss, predictions = loss_func(model, encoded_im, target)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step() 

		logger.step ("bits", binary.distance(predictions, target))
		#logger.step ("after", loss_func(model, image, target))

		if (i+1) % 400 == 0:
			test_transforms(model)
	


