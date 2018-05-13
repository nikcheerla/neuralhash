
import numpy as np
import random, sys, os, json, glob

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

import matplotlib.pyplot as plt
import IPython


def p(x):
	x = transforms.resize_rect(x)
	x = transforms.rotate(transforms.scale(x), max_angle=90)
	x = transforms.resize(x, rand_val=False, resize_val=224)
	x = transforms.translate(x)
	x = transforms.gauss(x, min_sigma=0.8, max_sigma=1.2)
	return x

def loss(model, image_batch):

	features = torch.cat([model(x, distribution=p, n=8).unsqueeze(0) \
		for x in image_batch], dim=0)
	correlations = corrcoef(features.t())
	correlations = correlations * (Variable(1 - torch.eye(TARGET_SIZE)).cuda())
	correlations = torch.pow(correlations, 2)
	print ("Mean: ", correlations.data.mean())
	print ("Max: ", correlations.data.max())
	return (correlations.mean()+correlations.max())


if __name__ == "__main__":
	
	model = DecodingNet()
	optimizer = torch.optim.Adadelta(model.parameters(), lr=0.003)
	
	def data_generator():
		files = glob.glob("/data/cats/*.jpg")
		for image in random.sample(files, k=len(files)):
			yield im.torch(im.load(image))

	for epochs in range(0, 150):
		for i, image_batch in enumerate(batched(data_generator(), batch_size=16)):

			error = loss(model, image_batch)
			print ("Epoch {0}, Batch {1}, Loss {2:0.5f}".format(epochs, 
				i, error.data.cpu().numpy().mean()))
			optimizer.zero_grad()
			error.backward()
			optimizer.step()

		if epochs < 2: continue
		model.save("/output/decorrelation.pth")

		target = binary.random(n=TARGET_SIZE)
		
		img = encode_binary(im.load("images/cat.jpg"), model, target=target, verbose=True)
		im.save(img, "/output/cat_encoded.jpg")

		preds = binary.get(model(im.torch(img)))
		print("Target: ", binary.str(target))
		print ("Code: ", binary.str(preds))
		print ("Diff: ", binary.distance(target, preds))

	model.save("/output/decorrelation.pth")


