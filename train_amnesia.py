
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
from logger import Logger

from skimage.morphology import binary_dilation
import IPython

from testing import test_transforms


DATA_PATH = 'data/amnesia'
logger = Logger("train", ("bce", "bits"), print_every=20)
EPSILON = 8e-3

def p(x):
    x = transforms.resize_rect(x)
    x = transforms.rotate(transforms.scale(x, 0.6, 1.4), max_angle=30)
    x = transforms.gauss(x, min_sigma=0.8, max_sigma=1.2)
    x = transforms.translate(x)
    x = transforms.identity(x)
    return x

def loss_func(model, encoded_im, target):
	scores = model.forward(encoded_im, distribution=p, n=64) # (N, T)
	predictions = scores.mean(dim=0)
	bce_loss = F.binary_cross_entropy(scores, \
		binary.target(target).repeat(64, 1))

	logger.step("bce", bce_loss)

	return bce_loss, predictions.cpu().data.numpy().round(2)

def init_data(input_path, output_path, n=100):
	os.system(f'rm {output_path}/*.pth')
	files = glob.glob(f'{input_path}/*.jpg')
	for k in range(n):
		img = im.load(random.choice(files))
		if img is None: continue
		img = im.torch(img).detach()
		perturbation = nn.Parameter(0.03*torch.randn(img.size()).to(DEVICE)+0.0).detach()
		target = binary.random(n=TARGET_SIZE)
		torch.save((perturbation, img, target, k), f'{output_path}/{target}_{k}.pth')

def save_data(input_path, output_path):
	files = glob.glob(f'{input_path}/*.pth')
	for file in files:
		perturbation, image, target = torch.load(random.choice(files))

		perturbation_zc = perturbation/perturbation.norm(2)*EPSILON*(perturbation.nelement()**0.5)
		changed_image = (image + perturbation_zc).clamp(min=0, max=1)

		im.save(im.numpy(image), file=f"{output_path}special/orig_{target}.jpg")
		im.save(im.numpy(changed_image), file=f"{output_path}changed_{target}.jpg")

if __name__ == "__main__":	

	model = DecodingNet()
	model.train()
	optimizer = torch.optim.Adadelta(model.classifier.parameters(), lr=1.5e-2)
	
	init_data('data/colornet', DATA_PATH, n=5000)

	def data_generator():
		# path = "/home/RC/neuralhash/data/tiny-imagenet-200/test/images"
		path = f"{DATA_PATH}/*.pth"
		files = glob.glob(path)
		while True:
			yield torch.load(random.choice(files))
			# if img is None: continue
			# yield img

	def checkpoint():
		print (f"Saving model to {OUTPUT_DIR}train_test.pth")
		# model.save(OUTPUT_DIR + "train_test.pth")

	logger.add_hook(checkpoint)

	for i in range(0, 10000):

		perturbation, orig_image, target, k = next(data_generator())
		perturbation.requires_grad = True

		encoded_im, new_perturbation = encode_binary(orig_image, model, \
		 	target, verbose=False, max_iter=1, perturbation=perturbation)
		encoded_im = im.torch(encoded_im)

		loss, predictions = loss_func(model, encoded_im, target)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		logger.step ("bits", binary.distance(predictions, target))
		#logger.step ("after", loss_func(model, image, target))

		#save encoded_im, target and perturbation
		torch.save((new_perturbation.detach(), orig_image, target, k), f'{DATA_PATH}/{target}_{k}.pth')

		if (i+1) % 400 == 0:
			test_transforms(model, images=[random.choice(os.listdir('data/colornet/')[:500])])
			# save_data(DATA_PATH, OUTPUT_DIR)

	# test_transforms(model, images=os.listdir(OUTPUT_DIR+"special/")[0:1])
