
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
from models import Discriminator
from logger import Logger

from skimage.morphology import binary_dilation
import IPython

DATA_PATH = 'data/encode_120'
logger = Logger("train_dsc", ("loss", "acc"), print_every=5, plot_every=20)

def loss_func(model, x, y):
	N = y.size(0)
	scores = model.forward(x)
	acc = np.sum([1 if scores[i,y[i]] > 0.5 else 0 for i in range(N)]) / N
	return F.cross_entropy(scores, y), acc

def data_gen(files, batch_size=128):
	while True:
		enc_files = random.sample(files, batch_size)
		orig_files = [f.replace('encoded', 'original') for f in enc_files]
		y = torch.zeros(batch_size, dtype=torch.long, device=DEVICE)
		x = []
		for i in range(batch_size):
			y[i] = random.randint(0, 1)
			if y[i] == 0:
				x.extend([im.load(enc_files[i]), im.load(orig_files[i])])
			else:
				x.extend([im.load(orig_files[i]), im.load(enc_files[i])])
		x = torch.stack([im.torch(img) for img in x])
		yield x, y

if __name__ == "__main__":

	model = nn.DataParallel(Discriminator())
	model.train()
	params = itertools.chain(model.module.classifier.parameters(), 
						model.module.features[-3:].parameters())
	optimizer = torch.optim.Adam(params, lr=1e-3)

	logger.add_hook(lambda: 
		[print (f"Saving model to {OUTPUT_DIR}train_dsc.pth"),
		model.module.save(OUTPUT_DIR + "train_dsc.pth")],
		freq=100,
	)

	files = glob.glob(f"{DATA_PATH}/*encoded*.jpg")
	train_files, val_files = files[:-128], files[-128:]
	x_val, y_val = next(data_gen(val_files, 128))

	for i, (x, y) in enumerate(data_gen(train_files, 128)):
		loss, acc = loss_func(model, x, y)
		# print(f'loss: {loss.data.cpu().numpy()}')
		logger.step ("loss", loss)
		logger.step ("acc", acc)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if i % 20 == 0:
			val_loss, val_acc = loss_func(model, x_val, y_val)
			print(f'val_loss = {val_loss}, val_acc = {val_acc}')

		if i == 400: break

