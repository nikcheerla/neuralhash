
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

	for k, files in tqdm.tqdm(list(enumerate(
						batch(glob.glob(f'{input_path}/*.jpg'), batch_size=64))), 
					ncols=50):

		images = im.stack([im.load(img_file) for img_file in files]).detach()
		perturbation = nn.Parameter(0.03*torch.randn(images.size()).to(DEVICE)+0.0)
		targets = [binary.random(n=TARGET_SIZE) for i in range(len(images))]
		torch.save((perturbation.data, images.data, targets), f'{output_path}/{k}.pth')

if __name__ == "__main__":	

	model = nn.DataParallel(DecodingNet(n=48, distribution=transforms.encoding))
	# params = itertools.chain(model.module.classifier.parameters(), 
	# 						model.module.features[-1].parameters())
	optimizer = torch.optim.Adam(model.module.classifier.parameters(), lr=2.5e-3)

	init_data('data/colornet', DATA_PATH, n=5000)

	logger.add_hook(lambda: 
		[print (f"Saving model to {OUTPUT_DIR}train_test.pth"),
		model.module.save(OUTPUT_DIR + "train_test.pth")],
		freq=40,
	)

	files = glob.glob(f"{DATA_PATH}/*.pth")
	for i, save_file in enumerate(random.choice(files) for i in range(0, 600)):

		perturbation, images, targets = torch.load(save_file)
		perturbation.requires_grad = True
		encoded_ims, perturbation = encode_binary(images, targets, \
			model, verbose=False, max_iter=1, perturbation=perturbation)

		loss, predictions = loss_func(model, encoded_ims, targets)
		logger.step ("loss", loss)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		torch.save((perturbation.data, images.data, targets), save_file)

		error = np.mean([binary.distance(x, y) for x, y in zip(predictions, targets)])
		logger.step ("bits", error)

		if i != 0 and i % 100 == 0:
			test_transforms(model, name=f'iter{i}')

