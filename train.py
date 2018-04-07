
import numpy as np
import random, sys, os, json, glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import *
from transforms import *
from models import DecodingNet

from skimage.morphology import binary_dilation

import matplotlib.pyplot as plt
import IPython

		# predictions = self.forward(x) # (BATCH_SIZE x 20)
		# predictions = predictions.t() # (20 x BATCH_SIZE)
		# correlations = corrcoef(predictions) # (20 x 20)
		# return correlations.sum()

model = DecodingNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

def loss(model, image_batch):
	features = torch.cat([model(x).unsqueeze(0) for x in image_batch], dim=0)
	print (features.size())
	correlations = corrcoef(features.t())
	correlations = torch.pow(correlations, 2)
	correlations = correlations.clamp(min=0.3, max=1.0)
	print (correlations)
	return correlations.mean()

def data_generator():
	for image in glob.glob("/data/cats/*.jpg"):
		yield im.torch(im.load(image))

for epochs in range(0, 500):
	for i, image_batch in enumerate(batched(data_generator(), batch_size=32)):

		error = loss(model, image_batch)
		print ("Epoch {0}, Batch {1}, Loss {2:0.5f}".format(epochs, 
			i, error.data.cpu().numpy().mean()))
		optimizer.zero_grad()
		error.backward()
		optimizer.step()

	model.save("/output/decorrelation.pth")
model.save("/output/decorrelation.pth")
