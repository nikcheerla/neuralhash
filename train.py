
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
	correlations = corrcoef(features.t())
	return correlations.sum()

def data_generator():
	for image in glob.glob("/data/cats/*.jpg"):
		yield im.torch(im.load(image))

for image_batch in batched(data_generator(), batch_size=32):

	loss = loss(model, image_batch)
	print ("Loss: ", loss.data)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()


