
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
	x = transforms.rotate(transforms.scale(x), max_angle=90)
	x = transforms.resize(x, rand_val=False, resize_val=224)
	x = transforms.translate(x)
	x = transforms.gauss(x, min_sigma=0.8, max_sigma=1.2)
	return x

def loss_func(model, encoded_image, target):
    predictions = model.forward(im.torch(encoded_image), distribution=p, n=64, return_variance=False) # (N, T)
    return F.binary_cross_entropy(predictions, binary.target(target).repeat(64, 1))

if __name__ == "__main__":

	model = DecodingNet()
	optimizer = torch.optim.SGD(model.features.classifier.parameters(), lr=1e-2)
	
	def data_generator():
		path = "/home/RC/neuralhash/data/tiny-imagenet-200/test/images"
		files = glob.glob(f"{path}/*.JPEG")
		for image in random.sample(files, k=len(files)):
			img = im.load(image)
			if img is None:
				continue
			yield img

	losses = []
	for i in range(0, 50):
		image = next(data_generator())
		target = binary.random(n=TARGET_SIZE)
		encoded_im = encode_binary(image, model, target, max_iter=10, verbose=False)
		loss = loss_func(model, encoded_im, target)
		losses.append(loss.cpu().data.numpy())

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if i % 5 == 0:
			model.drawLastLayer(OUTPUT_DIR + "mat_viz_" + str(i) + ".png")
		print("train loss = ", np.mean(losses[-1]))
		# print("loss after step = ", loss_func(model, encoded_im, target).cpu().data.numpy())

	test_transforms(model)
	model.save(OUTPUT_DIR + "train_test.pth")


