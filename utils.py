#utils.py

import numpy as np
import random, sys, os, time, glob

from skimage import io, color

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random

# CRITICAL HYPER PARAMS
EPSILON = 9e-3
BATCH_SIZE = 64
DIST_SIZE = 64
ENCODING_DIST_SIZE = 96
TARGET_SIZE = 32
VAL_SIZE = 8
P_RESET = 0 # prob that a encoded image is reset

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGE_MAX = 255.0
OUTPUT_DIR = "output/"
DATA_FILES = sorted(glob.glob("data/colornet/*.jpg"))
TRAIN_FILES, VAL_FILES = DATA_FILES[:5000], DATA_FILES[5000:5000+VAL_SIZE]


def corrcoef(x):
	mean_x = torch.mean(x, 1).unsqueeze(1)
	xm = x.sub(mean_x.expand_as(x))
	c = xm.mm(xm.t())
	c = c / (x.size(1) - 1)

	# normalize covariance matrix
	d = torch.diag(c)
	stddev = torch.pow(d, 0.5)
	c = c.div(stddev.expand_as(c))
	c = c.div(stddev.expand_as(c).t())

	# clamp between -1 and 1
	# probably not necessary but numpy does it
	c = torch.clamp(c, -1.0, 1.0)

	return c

def zca(x):
	sigma = torch.mm(x.t(), x) / x.shape[0]
	U, S, _ = torch.svd(sigma)
	pcs = torch.mm(torch.mm(U, torch.diag(1. / torch.sqrt(S + 1e-7))), U.t())

	# Apply ZCA whitening
	whitex = torch.mm(x, pcs)
	return whitex

def color_normalize(x):
	return torch.cat([((x[0]-0.485)/(0.229)).unsqueeze(0),
            ((x[1]-0.456)/(0.224)).unsqueeze(0),
            ((x[2]-0.406)/(0.225)).unsqueeze(0)], dim=0)

def tve_loss(x):
	return ((x[:,:-1,:] - x[:,1:,:])**2).sum() + ((x[:,:,:-1] - x[:,:,1:])**2).sum()

def batch(datagen, batch_size=32):
	arr = []
	for data in datagen:
		arr.append(data)
		if len(arr) == batch_size:
			yield arr
			arr = []
	yield arr

def batched(datagen, batch_size=32):
	arr = []
	for data in datagen:
		arr.append(data)
		if len(arr) == batch_size:
			yield list(zip(*arr))
			arr = []
	yield list(zip(*arr))

def elapsed(times=[time.time()]):
	times.append(time.time())
	return times[-1] - times[-2]

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

"""Image manipulation methods"""
class im(object):

	@staticmethod
	def load(filename):

		img = None

		try:
			img = io.imread(filename)/IMAGE_MAX
			if len(img.shape) != 3: return None
			img = img[:, :, 0:3]
		except (IndexError, OSError) as e:
			img = None

		return img

	@staticmethod
	def save(image, file="out.jpg"):
		io.imsave(file, (image*IMAGE_MAX).astype(np.uint8))

	@staticmethod
	def torch(image):
		x = torch.tensor(image).float().permute(2, 0, 1)
		return x.to(DEVICE)

	@staticmethod
	def numpy(image):
		return image.data.permute(1, 2, 0).cpu().numpy()

	@staticmethod
	def stack(images):
		return torch.cat([im.torch(image).unsqueeze(0) for image in images], dim=0)


"""Binary array data manipulation methods"""
class binary(object):

	@staticmethod
	def parse(bstr):
		return [int(c) for c in bstr]

	@staticmethod
	def get(predictions):
		if predictions is Variable:
			predictions = predictions.data.cpu().numpy()
		return list(predictions.clip(min=0, max=1).round().astype(int))

	@staticmethod
	def str(vals):
		return "".join([str(x) for x in vals])

	@staticmethod
	def target(values):
		values = torch.tensor(values).float()
		return values.to(DEVICE)

	@staticmethod
	def redundant(values, n=3):
		return list(values)*n

	@staticmethod
	def consensus(values, n=3):
		return list((np.reshape(values, (n, -1)).mean(axis=0) >= 0.5).astype(int))

	@staticmethod
	def random(n=10):
		return [random.randint(0, 1) for i in range(0, n)]

	@staticmethod
	def distance(code1, code2):
		code1 = np.array(code1).clip(min=0, max=1).round()
		code2 = np.array(code2).clip(min=0, max=1).round()
		num = 0
		for i in range(len(code1)):
			if code1[i] != code2[i]: num += 1
		return num

	@staticmethod
	def mse_dist(code1, code2):
		a = np.array(code1)
		b = np.array(code2)
		return np.mean((a - b)**2)

if __name__ == "__main__":

	data = im.load("test_data/n02108915_4657.jpg")
	im.save(data, file="out.jpg")

	print (im.torch(data).size())
	print (im.numpy(im.torch(data)).shape)
	im.save (im.numpy(im.torch(data)), file="out2.jpg")

	print (binary.consensus(binary.redundant([1, 1, 0, 1, 0, 0])))
	print (binary("111011"))
