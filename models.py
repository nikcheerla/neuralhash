
from __future__ import print_function

import numpy as np
import random, sys, os, json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchvision import models
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

	def set_distribution(self, distribution):
		self.distribution = distribution

	def forward(self, x):

		x = torch.cat([self.distribution(x).unsqueeze(1) \
						for i in range(0, self.n)], dim=1)
		B, N, C, H, W = x.shape

		x = torch.cat([((x[:, :, 0]-0.485)/(0.229)).unsqueeze(2),
			((x[:, :, 1]-0.456)/(0.224)).unsqueeze(2),
			((x[:, :, 2]-0.406)/(0.225)).unsqueeze(2)], dim=2)

		x = x.view(B*N, C, H, W)
		x = self.features(x)

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





"""Decoding network that tries to predict on a parallel batch"""
class DecodingGramNet(nn.Module):

	def __init__(self, distribution=transforms.identity, n=1):
		super(DecodingGramNet, self).__init__()

		self.features = models.squeezenet1_1(pretrained=True).features
		self.gram_classifiers = nn.ModuleList([
			nn.Linear(256**2, 256),
			nn.Linear(384**2, 256),
			nn.Linear(512**2, 256),
			])
		self.indices = [8, 10, 12]
		self.classifier = nn.Linear(256*3, TARGET_SIZE*2)
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

		layers = list(self.features._modules.values())
		gram_maps = []

		for i, layer in enumerate(layers):
			x = layer(x)

			j = self.indices.index(i) if i in self.indices else None

			if j is not None:
				y = gram(x).view(x.shape[0], -1)
				z = self.gram_classifiers[j](y)
				gram_maps.append(z)

			print (i, x.shape)

		# gram_maps = []
		# for layer, clf in zip(layers[-3:], self.gram_classifiers):
		# 	x = layer(x)
		# 	y = gram(x).view(x.shape[0], -1)
		# 	print (x.shape, y.shape)
		# 	print (clf)
		# 	#gram_maps.append(clf(y))

		x = torch.cat(gram_maps, dim=1)

		x = (x - x.mean(dim=1, keepdim=True))/(x.std(dim=1, keepdim=True))
		x = self.classifier(x)
		x = x.view(B, N, TARGET_SIZE, 2)#.mean(dim=0) # reshape and average

		return F.softmax(x, dim=3)[:,:, :, 0].clamp(min=0, max=1)


	def load(self, file_path):
		self.load_state_dict(torch.load(file_path))

	def save(self, file_path):
		torch.save(self.state_dict(), file_path)







"""CNN that discriminates between encoded im and original im"""
class Discriminator(nn.Module):
	
	def __init__(self):
		super(Discriminator, self).__init__()

		self.features = models.vgg11(pretrained=True).features
		self.classifier = nn.Sequential(
			nn.Linear(50176, 1000),
			nn.ReLU(),
			nn.BatchNorm1d(1000),
			nn.Linear(1000, 2))
		self.to(DEVICE)

	def forward(self, x):
		N, C, H, W = x.shape
		x = transforms.identity(x)

		x = torch.cat([((x[:, 0]-0.485)/(0.229)).unsqueeze(1),
			((x[:, 1]-0.456)/(0.224)).unsqueeze(1),
			((x[:, 2]-0.406)/(0.225)).unsqueeze(1)], dim=1)
		
		x = self.features(x)
		x = self.classifier(x.view(N//2, -1))

		return F.softmax(x, dim=1)
		
	def load(self, file_path):
		self.load_state_dict(torch.load(file_path))

	def save(self, file_path):
		torch.save(self.state_dict(), file_path)


class UNet_down_block(nn.Module):
	def __init__(self, input_channel, output_channel, down_size=True):
		super(UNet_down_block, self).__init__()
		self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=1)
		self.bn1 = nn.BatchNorm2d(output_channel)
		self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(output_channel)
		self.max_pool = nn.MaxPool2d(2, 2)
		self.relu = nn.ReLU()
		self.down_size = down_size

	def forward(self, x):

		x = self.relu(self.bn1(self.conv1(x)))
		x = self.relu(self.bn2(self.conv2(x)))
		if self.down_size:
			x = self.max_pool(x)
		return x

class UNet_up_block(nn.Module):
	def __init__(self, prev_channel, input_channel, output_channel, up_sample=True):
		super(UNet_up_block, self).__init__()
		self.up_sampling = nn.Upsample(scale_factor=2, mode='bilinear')
		self.conv1 = nn.Conv2d(prev_channel + input_channel, output_channel, 3, padding=1)
		self.bn1 = nn.BatchNorm2d(output_channel)
		self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(output_channel)
		self.relu = torch.nn.ReLU()
		self.up_sample = up_sample

	def forward(self, prev_feature_map, x):
		if self.up_sample:
			x = self.up_sampling(x)
		x = torch.cat((x, prev_feature_map), dim=1)
		x = self.relu(self.bn1(self.conv1(x)))
		x = self.relu(self.bn2(self.conv2(x)))
		return x

class UNet(torch.nn.Module):
	def __init__(self):
		super(UNet, self).__init__()

		self.down_block1 = UNet_down_block(3, 16, False)
		self.down_block2 = UNet_down_block(16, 32, True)
		self.down_block3 = UNet_down_block(32, 64, True)
		self.down_block4 = UNet_down_block(64, 128, True)
		self.down_block5 = UNet_down_block(128, 256, True)
		# self.down_block6 = UNet_down_block(256, 512, True)

		self.mid_conv1 = nn.Conv2d(256, 256, 3, padding=1)
		self.bn1 = nn.BatchNorm2d(256)
		# self.mid_conv2 = nn.Conv2d(512, 512, 3, padding=1)
		# self.bn2 = nn.BatchNorm2d(512)
		# self.mid_conv3 = torch.nn.Conv2d(512, 512, 3, padding=1)
		# self.bn3 = torch.nn.BatchNorm2d(512)

		# self.up_block1 = UNet_up_block(256, 512, 256)
		self.up_block2 = UNet_up_block(128, 256, 128)
		self.up_block3 = UNet_up_block(64, 128, 64)
		self.up_block4 = UNet_up_block(32, 64, 32)
		self.up_block5 = UNet_up_block(16, 32, 16)

		self.last_conv1 = nn.Conv2d(16, 16, 3, padding=1)
		self.last_bn = nn.BatchNorm2d(16)
		self.last_conv2 = nn.Conv2d(16, 3, 1, padding=0)
		self.relu = nn.ReLU()
		self.to(DEVICE)

	def forward(self, x):
		self.x1 = self.down_block1(x)
		self.x2 = self.down_block2(self.x1)
		self.x3 = self.down_block3(self.x2)
		self.x4 = self.down_block4(self.x3)
		self.x5 = self.down_block5(self.x4)
		# self.x6 = self.down_block6(self.x5)

		self.x5 = self.relu(self.bn1(self.mid_conv1(self.x5)))
		# self.x6 = self.relu(self.bn2(self.mid_conv2(self.x6)))
		# self.x6 = self.relu(self.bn3(self.mid_conv3(self.x6)))

		# x = self.up_block1(self.x5, self.x6)
		x = self.up_block2(self.x4, self.x5)
		x = self.up_block3(self.x3, x)
		x = self.up_block4(self.x2, x)
		x = self.up_block5(self.x1, x)
		x = self.relu(self.last_bn(self.last_conv1(x)))
		x = self.last_conv2(x)
		return x, [self.x1, self.x2, self.x3, self.x4, self.x5]

	def load(self, file_path):
		self.load_state_dict(torch.load(file_path))

	def save(self, file_path):
		torch.save(self.state_dict(), file_path)


"""Decoding network that tries to predict a
binary value of size target_size """
class DilatedDecodingNet(nn.Module):

	def __init__(self):
		super(DilatedDecodingNet, self).__init__()

		self.features = models.vgg11(pretrained=True)
		self.classifier = nn.Linear(512**2, TARGET_SIZE*2)
		self.gram = GramMatrix()
		self.features.eval()

		if USE_CUDA: self.cuda()

	def forward(self, x, verbose=False, distribution=transforms.identity, 
					n=1, return_variance=False):

		# make sure to center the image and divide by standard deviation
		x = torch.cat([((x[0]-0.485)/(0.229)).unsqueeze(0),
			((x[1]-0.456)/(0.224)).unsqueeze(0),
			((x[2]-0.406)/(0.225)).unsqueeze(0)], dim=0)

		x = torch.cat([distribution(x).unsqueeze(0) for i in range(0, n)], dim=0)

		#vgg layers
		dilation_factor = 1
		for layer in list(self.features.features._modules.values()):
			if isinstance(layer, nn.Conv2d):
				x = F.conv2d(x, layer.weight, bias=layer.bias, stride=layer.stride, \
					padding=tuple(layer.padding*np.array(dilation_factor)), dilation=dilation_factor)
			elif isinstance(layer, nn.MaxPool2d):
				if dilation_factor == 1:
					x = F.max_pool2d(x, 2, stride=1, dilation=1)
					x = F.pad(x, (1, 0, 1, 0))
				else:
					x = F.max_pool2d(x, 2, stride=1, dilation=dilation_factor)
					x = F.pad(x, [dilation_factor//2]*4)
				dilation_factor *= 2
			else:
				x = layer(x)

		x = self.gram(x)
		x = x.view(x.size(0), -1)
		x = (x - x.mean(dim=1, keepdim=True))/(x.std(dim=1, keepdim=True))
		x = self.classifier(x)
		x = x.view(x.size(0), TARGET_SIZE, 2)#.mean(dim=0) # reshape and average

		predictions = F.softmax(x, dim=2)[:,:, 0]

		return predictions

	def drawLastLayer(self, file_path):
		img = self.classifier.weight.cpu().data.numpy()
		plt.imshow(img, cmap='hot')
		plt.savefig(file_path)

	def load(self, file_path):
		self.load_state_dict(torch.load(file_path))

	def save(self, file_path):
		torch.save(self.state_dict(), file_path)


if __name__ == "__main__":

	model = nn.DataParallel(DecodingGramNet(n=1, distribution=transforms.identity))
	images = torch.randn(1, 3, 224, 224).float().to(DEVICE)
	x = model.forward(images)
	print (x.shape)
