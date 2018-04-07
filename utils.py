
import numpy as np
import random, sys, os

from skimage import io, color

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random


USE_CUDA = torch.cuda.is_available()
IMAGE_MAX = 255.0
TARGET_SIZE = 32


def corrcoef(x):
    mean_x = torch.mean(x, 1)
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


def batched(datagen, batch_size=32):
	arr = []
	for data in datagen:
		arr.append(data)
		if len(arr) == batch_size:
			yield arr
			arr = []
	yield arr


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
		x = Variable(torch.FloatTensor(image).permute(2, 0, 1))
		if USE_CUDA: x = x.cuda()
		return x

	@staticmethod
	def numpy(image):
		return image.data.permute(1, 2, 0).cpu().numpy()


"""Binary array data manipulation methods"""
class binary(object):

	@staticmethod
	def parse(bstr):
		return [int(c) for c in bstr]

	@staticmethod
	def get(predictions):
		return list(predictions.data.cpu().numpy().clip(min=0, max=1).round().astype(int))

	@staticmethod
	def str(vals):
		return "".join([str(x) for x in vals])

	@staticmethod
	def target(values):
		values = Variable(torch.FloatTensor(np.array([float(x) for x in values])))
		if USE_CUDA: values = values.cuda()
		return values

	@staticmethod
	def redundant(values, n=3):
		return list(values)*3

	@staticmethod
	def consensus(values, n=3):
		return list((np.reshape(values, (n, -1)).mean(axis=0) >= 0.5).astype(int))

	@staticmethod
	def random(n=10):
		return [random.randint(0, 1) for i in range(0, n)]

	@staticmethod
	def distance(code1, code2):
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

