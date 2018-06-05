
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
logger = Logger("train", ("bce", "bits"), print_every=1)
EPSILON = 8e-3

def loss_func(model, x, targets):
	scores = model.forward(x)
	predictions = scores.mean(dim=1)
	score_targets = binary.target(targets).unsqueeze(1).expand_as(scores)

	return F.binary_cross_entropy(scores, score_targets), \
		predictions.cpu().data.numpy().round(2)

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

	def p(x):
		x = transforms.resize_rect(x)
		x = transforms.rotate(transforms.scale(x, 0.6, 1.4), max_angle=30)
		x = transforms.gauss(x, min_sigma=0.8, max_sigma=1.2)
		x = transforms.translate(x)
		x = transforms.identity(x)
		return x

	model = nn.DataParallel(DecodingNet(n=48, distribution=p))
	optimizer = torch.optim.Adadelta(model.module.classifier.parameters(), lr=8e-2)
	model.train()
	
	init_data('data/colornet', DATA_PATH, n=100)

	def data_generator():
		path = f"{DATA_PATH}/*.pth"
		files = glob.glob(path)
		while True:
			yield torch.load(random.choice(files))

	def checkpoint():
		print (f"Saving model to {OUTPUT_DIR}train_test.pth")
		# model.save(OUTPUT_DIR + "train_test.pth")

	logger.add_hook(checkpoint)

	for i, (perturbations, orig_images, targets, ks) in enumerate(batched(data_generator())):

		perturbations = torch.stack(perturbations)
		orig_images = torch.stack(orig_images)
		perturbations.requires_grad = True

		encoded_ims, new_perturbations = encode_binary(orig_images, targets, \
		 	model, verbose=False, max_iter=1, perturbation=perturbations)

		loss, predictions = loss_func(model, encoded_ims, targets)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		error = np.mean([binary.distance(x, y) for x, y in zip(predictions, targets)])
		logger.step ("bits", error)

		#save encoded_im, target and perturbation
		for new_p, orig_image, target, k in zip(new_perturbations, orig_images, targets, ks):
			torch.save((new_p, orig_image, target, k), f'{DATA_PATH}/{target}_{k}.pth')

		if (i+1) % 3 == 0:
			test_transforms(model, images=[random.choice(glob.glob('data/colornet/')[:500])])
			# save_data(DATA_PATH, OUTPUT_DIR)

	# test_transforms(model, images=os.listdir(OUTPUT_DIR+"special/")[0:1])
