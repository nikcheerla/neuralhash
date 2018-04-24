
import numpy as np
import random, sys, os, json, glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import *
import transforms
from models import DecodingNet

from skimage.morphology import binary_dilation

import matplotlib.pyplot as plt
import IPython


model = DecodingNet()
optimizer = torch.optim.Adadelta(model.parameters(), lr=0.003)

def loss(model, image_batch):

	def p(x):
		x = transforms.resize_rect(x)
		x = transforms.rotate(transforms.scale(x), max_angle=90)
		x = transforms.resize(x, rand_val=False, resize_val=224)
		x = transforms.translate(x)
		x = transforms.gauss(x, min_sigma=0.8, max_sigma=1.2)
		return x

	features = torch.cat([model(x, distribution=p, n=8).unsqueeze(0) \
		for x in image_batch], dim=0)
	correlations = corrcoef(features.t())
	correlations = correlations * (Variable(1 - torch.eye(TARGET_SIZE)).cuda())
	correlations = torch.pow(correlations, 2)

	print ("Covariance matrix: ", correlations.data.cpu().numpy())
	return (correlations.mean())



if __name__ == "__main__":
	
	model = DecodingNet()
	
	

	def data_generator():
		files = glob.glob("/data/cats/*.jpg")
		for image in random.sample(files, k=len(files)):
			yield im.torch(im.load(image))

	for epochs in range(0, 150):
		for i, image_batch in enumerate(batched(data_generator(), batch_size=32)):

			error = loss(model, image_batch)
			print ("Epoch {0}, Batch {1}, Loss {2:0.5f}".format(epochs, 
				i, error.data.cpu().numpy().mean()))
			optimizer.zero_grad()
			error.backward()
			optimizer.step()

		model.save("/output/decorrelation.pth")

		target = binary.random(n=TARGET_SIZE)
		print("Target: ", binary.str(target))
		encode_binary(im.load("images/cat.jpg"), model, target=target, verbose=True)
		
	model.save("/output/decorrelation.pth")


