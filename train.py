
import numpy as np
import random, sys, os, json, glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchvision import models, datasets, transforms
from utils import im, binary, corrcoef
from transforms import *

from skimage.morphology import binary_dilation

import matplotlib.pyplot as plt
import IPython


USE_CUDA = torch.cuda.is_available()
EPSILON = 3e-2
MIN_LOSS = 7e-2
BATCH_SIZE = 64

class DecodingNet(nn.Module):

	def __init__(self, target_size=10):
		super(DecodingNet, self).__init__()

		self.features = models.vgg11(pretrained=True)
		self.features.classifier = nn.Sequential(
			nn.Linear(25088, target_size))

		self.target_size=target_size
		self.features.eval()

		if USE_CUDA: self.cuda()

	def set_target(self, target):
		self.target=target

	def forward(self, x, verbose=False):

		# make sure to center the image and divide by standard deviation
		x = torch.cat([((x[:, 0]-0.485)/(0.229)).unsqueeze(1), 
			((x[:, 1]-0.456)/(0.224)).unsqueeze(1), 
			((x[:, 2]-0.406)/(0.225)).unsqueeze(1)], dim=1)

		# returns an image after a series of transformations
		def distribution(x):
			x = resize(x, rand_val=False, resize_val=224)
			return x

		images = x
		predictions = self.features(images) + 0.5

		return predictions.mean(dim=0)

	""" returns the accuracy loss as well as the predictions """
	def loss(self, x):
		predictions = self.forward(x) # (BATCH_SIZE x 20)
		predictions = predictions.t() # (20 x BATCH_SIZE)
		correlations = corrcoef(predictions) # (20 x 20)
		return correlations.sum()

model = DecodingNet(target_size=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

train_dataset = datasets.ImageFolder(
	"/data",
	transforms.Compose([
		transforms.Scale(224),
		transforms.RandomSizedCrop(224),
		transforms.ToTensor(),
	]))

print ("here1!")

train_loader = torch.utils.data.DataLoader(
	train_dataset, batch_size=BATCH_SIZE, shuffle=True,
	num_workers=4, pin_memory=True)

print ("here2!")
for i, (input_batch, target_batch) in enumerate(train_loader):
	print (i)
	#loss = model.loss(Variable(input_batch).cuda())
	#print ("Loss: ", loss.data)
	#optimizer.zero_grad()
	#loss.backward()
	optimizer.step()
