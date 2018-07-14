
from __future__ import print_function

import numpy as np
import random, sys, os, json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchvision import models
from modules import *
from utils import *
import transforms

import IPython




"""Decoding network that tries to predict on a parallel batch"""
class DecodingNet(nn.Module):

	def __init__(self, distribution=transforms.identity, n=1):
		super(DecodingNet, self).__init__()

		self.features = models.squeezenet1_1(pretrained=True).features
		self.classifier = nn.Sequential(
			nn.Linear(512*8, TARGET_SIZE*2),)
			#nn.ReLU(inplace=True),
			#nn.Linear(4096, TARGET_SIZE*2))
		self.bn = nn.BatchNorm2d(512)
		self.distribution, self.n = distribution, n
		self.to(DEVICE)

	def forward(self, x):

		x = torch.cat([self.distribution(x).unsqueeze(1) \
						for i in range(0, self.n)], dim=1)
		B, N, C, H, W = x.shape

		x = torch.cat([((x[:, :, 0]-0.485)/(0.229)).unsqueeze(2),
			((x[:, :, 1]-0.456)/(0.224)).unsqueeze(2),
			((x[:, :, 2]-0.406)/(0.225)).unsqueeze(2)], dim=2)

		x = x.view(B*N, C, H, W)

		x = self.features(x)
		#x = self.bn(x)

		x = torch.cat([F.avg_pool2d(x, (x.shape[2]//2)), \
						F.max_pool2d(x, (x.shape[2]//2))], dim=1)
		x = x.view(x.size(0), -1)
		x = (x - x.mean(dim=1, keepdim=True))/(x.std(dim=1, keepdim=True))
		x = self.classifier(x)
		x = x.view(B, N, TARGET_SIZE, 2)#.mean(dim=0) # reshape and average

		return F.softmax(x, dim=3)[:,:, :, 0].clamp(min=0, max=1)

	def load(self, file_path):
		self.load_state_dict(torch.load(file_path))

	def save(self, file_path):
		torch.save(self.state_dict(), file_path)



# """Decoding network that tries to predict a
# binary value of size target_size """
# class DilatedDecodingNet(nn.Module):

#     def __init__(self):
#         super(DilatedDecodingNet, self).__init__()

#         self.features = models.vgg11(pretrained=True)
#         self.classifier = nn.Linear(512**2, TARGET_SIZE*2)
#         self.gram = GramMatrix()
#         self.features.eval()

#         if USE_CUDA: self.cuda()

#     def forward(self, x, verbose=False, distribution=transforms.identity, 
#                     n=1, return_variance=False):

#         # make sure to center the image and divide by standard deviation
#         x = torch.cat([((x[0]-0.485)/(0.229)).unsqueeze(0),
#             ((x[1]-0.456)/(0.224)).unsqueeze(0),
#             ((x[2]-0.406)/(0.225)).unsqueeze(0)], dim=0)

#         x = torch.cat([distribution(x).unsqueeze(0) for i in range(0, n)], dim=0)

#         #vgg layers
#         dilation_factor = 1
#         for layer in list(self.features.features._modules.values()):
#             if isinstance(layer, nn.Conv2d):
#                 x = F.conv2d(x, layer.weight, bias=layer.bias, stride=layer.stride, \
#                     padding=tuple(layer.padding*np.array(dilation_factor)), dilation=dilation_factor)
#             elif isinstance(layer, nn.MaxPool2d):
#                 if dilation_factor == 1:
#                     x = F.max_pool2d(x, 2, stride=1, dilation=1)
#                     x = F.pad(x, (1, 0, 1, 0))
#                 else:
#                     x = F.max_pool2d(x, 2, stride=1, dilation=dilation_factor)
#                     x = F.pad(x, [dilation_factor//2]*4)
#                 dilation_factor *= 2
#             else:
#                 x = layer(x)

#         x = self.gram(x)
#         x = x.view(x.size(0), -1)
#         x = (x - x.mean(dim=1, keepdim=True))/(x.std(dim=1, keepdim=True))
#         x = self.classifier(x)
#         x = x.view(x.size(0), TARGET_SIZE, 2)#.mean(dim=0) # reshape and average

#         predictions = F.softmax(x, dim=2)[:,:, 0]

#         return predictions

#     def drawLastLayer(self, file_path):
#         img = self.classifier.weight.cpu().data.numpy()
#         plt.imshow(img, cmap='hot')
#         plt.savefig(file_path)

#     def load(self, file_path):
#         self.load_state_dict(torch.load(file_path))

#     def save(self, file_path):
#         torch.save(self.state_dict(), file_path)




if __name__ == "__main__":

	model = nn.DataParallel(DecodingNet(n=64, distribution=transforms.training))
	images = torch.randn(48, 3, 224, 224).float().to(DEVICE).requires_grad_()
	x = model.forward(images)

	x.mean().backward()
	print (x.shape)
