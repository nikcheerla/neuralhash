
import numpy as np
import random, sys, os, json, glob
import tqdm, itertools, shutil

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import torch
torch.backends.cudnn.benchmark=True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import *
import transforms
from encoding import encode_binary
from models import DecodingNet, DecodingGramNet
from logger import Logger

from skimage.morphology import binary_dilation
import IPython

from testing import test_transforms


logger = Logger("train", ("loss", "bits"), print_every=10, plot_every=40)

def loss_func(model, x, targets):
	scores = model.forward(x)
	predictions = scores.mean(dim=1)
	score_targets = binary.target(targets).unsqueeze(1).expand_as(scores)

	return F.binary_cross_entropy(scores, score_targets), \
		predictions.cpu().data.numpy().round(2)

def init_data(output_path, n=None):
	
	shutil.rmtree(output_path)
	os.makedirs(output_path)

	image_files = TRAIN_FILES
	if n is not None: image_files = image_files[0:n]

	for k, files in tqdm.tqdm(list(enumerate(
						batch(image_files, batch_size=BATCH_SIZE))), 
					ncols=50):

		images = im.stack([im.load(img_file) for img_file in files]).detach()
		perturbation = nn.Parameter(0.03*torch.randn(images.size()).to(DEVICE)+0.0)
		targets = [binary.random(n=TARGET_SIZE) for i in range(len(images))]
		optimizer = torch.optim.Adam([perturbation], lr=ENCODING_LR)
		torch.save((perturbation.data, images.data, targets, optimizer.state_dict()), f'{output_path}/{k}.pth')

if __name__ == "__main__":	

	model = nn.DataParallel(DecodingGramNet(n=DIST_SIZE, distribution=transforms.easy))
	# params = itertools.chain(model.module.gram_classifiers.parameters(), 
	# 						model.module.classifier.parameters())
	optimizer = torch.optim.Adam(model.module.classifier.parameters(), lr=2.5e-3)
	#init_data("data/amnesia")

	logger.add_hook(lambda: 
		[print (f"Saving model to {OUTPUT_DIR}train_test.pth"),
		model.module.save(OUTPUT_DIR + "train_test.pth")],
		freq=40,
	)

	files = glob.glob(f"data/amnesia/*.pth")
	for i, save_file in enumerate(random.choice(files) for i in range(0, 800)):

		perturbation, images, targets, optimizer_state = torch.load(save_file)
		perturbation = perturbation.requires_grad_()
		pert_optimizer = torch.optim.Adam([perturbation], lr=ENCODING_LR)
		pert_optimizer.load_state_dict(optimizer_state)

		perturbation.requires_grad = True
		encoded_ims, perturbation, pert_optimizer = encode_binary(images, targets, \
			model, verbose=False, max_iter=1, perturbation=perturbation, optimizer=pert_optimizer)

		loss, predictions = loss_func(model, encoded_ims, targets)
		logger.step ("loss", loss)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		torch.save((perturbation.data, images.data, targets, pert_optimizer.state_dict()), save_file)

		error = np.mean([binary.distance(x, y) for x, y in zip(predictions, targets)])
		logger.step ("bits", error)

		if i != 0 and i % 300 == 0:
			#test_transforms(model, random.sample(TRAIN_FILES, 16), name=f'iter{i}_train')
			test_transforms(model, VAL_FILES, name=f'iter{i}_test')

