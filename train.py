
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

def loss_func(model, encoded_im, target):
	predictions = model.forward(encoded_im, distribution=p, n=64) # (N, T)
	return F.binary_cross_entropy(predictions, binary.target(target).repeat(64, 1))

def create_targets():
	return [(i == j) for i, j in product(range(TARGET_SIZE), range(TARGET_SIZE))]

if __name__ == "__main__":

	model = DecodingNet()
	optimizer = torch.optim.Adadelta(model.classifier.parameters(), lr=1e-2)
	
	def data_generator():
		# path = "/home/RC/neuralhash/data/tiny-imagenet-200/test/images"
		path = "data/colornet/2vpu9L.jpg"
		files = glob.glob(path)
		while True:
			img = im.load(random.choice(files))
			if img is None:
				continue
			yield img

	losses = []

	
	target = binary.random(n=TARGET_SIZE)

	for i in range(0, 1000):

		image = next(data_generator())
		encoded_im = im.torch(encode_binary(image, model, target, max_iter=3))
		
		loss = loss_func(model, encoded_im, target)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step() 

		losses.append(loss.cpu().data.numpy())

		if i % 10 == 0:
			model.drawLastLayer(OUTPUT_DIR + "mat_viz_" + str(i) + ".png")

		print("train loss = ", losses[-1])
		print("loss after step = ", loss_func(model, encoded_im, target).cpu().data.numpy())

	test_transforms(model)
	model.save(OUTPUT_DIR + "train_test.pth")


