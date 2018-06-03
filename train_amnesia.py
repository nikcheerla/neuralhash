
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
import math 

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

	def amnesia_data_generator():
		path = "data/tiny-imagenet-200/train"
		files = glob.glob(f"{path}/*/images/*.pth")
		for image_file in files:
			img_tensor = torch.load(image_file)
			yield img_tensor

	losses = []

	remember_path = "data/tiny-imagenet-200/remember"

	for i in range(0, 50):
		print('i: ' + str(i))
		image = next(amnesia_data_generator())
		target = binary.random(n=TARGET_SIZE) #TARGET_SIZE
		encoded_im_numpy = encode_binary(image, model, target, max_iter=10, verbose=False)
		encoded_im_tensor = torch.from_numpy(encoded_im_numpy)
		target_str = ''.join(map(str, target))
		torch.save(encoded_im_tensor, remember_path + "/" + str(i) + "_" + target_str + ".pth")
		loss = loss_func(model, encoded_im_numpy, target)
		losses.append(loss.cpu().data.numpy())
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		#retrain on previous images to not forget their patterns
		#HYPERPARAMETERS
		N = 2 #number of images to sample
		p = 0.2 #proportion of previous images to sample from (i.e. 20% is)
		# for m in range(N):
		# 	index = math.ceil(i-random.random()*p*i)
		# 	pathname = glob.glob(remember_path + "/" + str(index) + "_*.pth")
		# 	img_tensor = torch.load(pathname[0]).to(DEVICE)
		# 	target = pathname[0][-36: -4]
		# 	encoded_img_numpy = encode_binary(img_tensor, model, target, max_iter=5, verbose=False)
		# 	loss = loss_func(model, encoded_img_numpy, target)
		# 	loss.backward()
		# optimizer.step()

		#pick 2
		img_1_index = math.ceil(i-random.random()*0.2*i)
		img_2_index = math.ceil(i-random.random()*0.2*i)
		#print('image indexes ' + str(img_1_index) + ', ' + str(img_2_index))
		glob1 = glob.glob(remember_path + "/" + str(img_1_index) + "_*.pth")
		glob2 = glob.glob(remember_path + "/" + str(img_1_index) + "_*.pth")
		img1_tensor = torch.load(glob1[0]).to(DEVICE)
		img2_tensor = torch.load(glob1[0]).to(DEVICE)
		target_img1 = glob1[0][-36: -4] #retrieves target from filename
		target_img2 = glob2[0][-36: -4] 
		print('here')
		encoded_img1_numpy = encode_binary(img1_tensor, model, target_img1, max_iter=5, verbose=False)
		encoded_img2_numpy = encode_binary(img2_tensor, model, target_img2, max_iter=5, verbose=False)
		loss1 = loss_func(model, encoded_img1_numpy, target_img1)	
		loss2 = loss_func(model, encoded_img2_numpy, target_img2)	
		
		optimizer.zero_grad()
		loss1.backward()
		#loss2.backward()
		optimizer.step()
		
		#encode_binary(img_1, model, target, max_iter=10, vebose=False)

		if i % 10 == 0:
			model.drawLastLayer(OUTPUT_DIR + "mat_viz_adam2_" + str(i) + ".png")
		print("train loss = ", np.mean(losses[-1]))
		# print("loss after step = ", loss_func(model, encoded_im, target).cpu().data.numpy())

	test_transforms(model)
	model.save(OUTPUT_DIR + "train_test.pth")


