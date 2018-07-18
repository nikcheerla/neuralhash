
import numpy as np
import random, sys, os, json, glob
import tqdm, itertools, shutil

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
logger = Logger("train", ("loss", "bits"), print_every=5, plot_every=20)

def loss_func(model, x, targets):
	scores = model.forward(x)
	predictions = scores.mean(dim=1)
	score_targets = binary.target(targets).unsqueeze(1).expand_as(scores)

	return F.binary_cross_entropy(scores, score_targets), \
		predictions.cpu().data.numpy().round(2)

def init_data(input_path, output_path, n=100):
	shutil.rmtree(DATA_PATH)
	os.makedirs(DATA_PATH)
	files = glob.glob(f'{input_path}/*.jpg')


	for k, img_file in tqdm.tqdm(list(enumerate(files)), ncols=50):
		img = im.load(img_file)
		if img is None: continue
		img = im.torch(img).detach()
		perturbation = nn.Parameter(0.03*torch.randn(img.size()).to(DEVICE)+0.0).detach()
		target = binary.random(n=TARGET_SIZE)
		torch.save((perturbation, img, target, k), f'{output_path}/{k}.pth')

if __name__ == "__main__":	

	model = nn.DataParallel(DecodingNet(n=48, distribution=transforms.encoding))
	# params = itertools.chain(model.module.classifier.parameters(), 
	# 						model.module.features[-1].parameters())
	optimizer = torch.optim.Adam(model.module.classifier.parameters(), lr=2.5e-3)

	#model.train()
	
	init_data('data/colornet', DATA_PATH)

	def data_generator():
		path = f"{DATA_PATH}/*.pth"
		files = glob.glob(path)  #    <--- TODO don't redo every time
		while True:
			yield torch.load(random.choice(files))

	def checkpoint():
		print (f"Saving model to {OUTPUT_DIR}train_test.pth")
		model.module.save(OUTPUT_DIR + "train_test.pth")

	logger.add_hook(checkpoint)

	for i, (perturbations, orig_images, targets, ks) in \
			enumerate(batched(data_generator(), batch_size=BATCH_SIZE)):

		perturbations = torch.stack(perturbations)
		orig_images = torch.stack(orig_images)
		perturbations.requires_grad = True

		encoded_ims, new_perturbations = encode_binary(orig_images, targets, \
			model, verbose=False, max_iter=1, perturbation=perturbations)

		loss, predictions = loss_func(model, encoded_ims, targets)
		logger.step ("loss", loss)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		error = np.mean([binary.distance(x, y) for x, y in zip(predictions, targets)])
		logger.step ("bits", error)

		# #save encoded_im, target and perturbation
		# for new_p, orig_image, target, k in zip(new_perturbations, orig_images, targets, ks):
		# 	if random.random() < P_RESET:
		# 		os.remove(f'{DATA_PATH}/{k}.pth')
		# 		new_p = nn.Parameter(0.03*torch.randn(orig_image.size()).to(DEVICE)+0.0).detach()
		# 		target = binary.random(n=TARGET_SIZE)
		# 		torch.save((torch.tensor(new_p.data), 
		# 					torch.tensor(orig_image.data), target, k), 
		# 					f'{DATA_PATH}/{k}.pth')

		if i != 0 and i % 100 == 0:
			test_transforms(model, name=f'iter{i}')
	
		if i == 600:
			break

