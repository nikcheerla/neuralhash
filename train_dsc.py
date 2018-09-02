
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
from modules import UNet
from logger import Logger

from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import IPython

DATA_PATH = 'jobs/experiment59/output/'
logger = Logger("train_dsc", ("loss", "corr"), print_every=5, plot_every=20)

def loss_func(model, x, y):
	cleaned = model.forward(x)
	corr, p = pearsonr(cleaned.data.cpu().numpy().flatten(), y.data.cpu().numpy().flatten())
	return (cleaned - y).pow(2).sum(), corr

def data_gen(files):
	while True:
		batch = random.choice(files)
		perturbation, images, targets = torch.load(batch)        
		perturbation_zc = perturbation/perturbation.view(perturbation.shape[0], -1)\
			.norm(2, dim=1, keepdim=True).unsqueeze(2).unsqueeze(2).expand_as(perturbation)\
			*EPSILON*(perturbation[0].nelement()**0.5)
		enc_ims = (images + perturbation_zc).clamp(min=0.0, max=1.0)
		yield enc_ims, perturbation_zc

def viz_preds(model, x, y):
	preds = model(x)
	for i, (pred, truth, enc) in enumerate(zip(preds, y, x)):
		im.save(im.numpy(enc), f'{OUTPUT_DIR}{i}_encoded.jpg')
		im.save(3*np.abs(im.numpy(pred)), f'{OUTPUT_DIR}{i}_pred.jpg')
		im.save(3*np.abs(im.numpy(truth)), f'{OUTPUT_DIR}{i}_truth.jpg')

if __name__ == "__main__":

	model = nn.DataParallel(UNet())
	model.train()

	optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

	# optimizer.load_state_dict('output/unet_opt.pth')	
	# model.module.load('jobs/experiment_unet/output/train_unet.pth')

	logger.add_hook(lambda: 
		[print (f"Saving model/opt to {OUTPUT_DIR}train_unet.pth"),
		model.module.save(OUTPUT_DIR + "train_unet.pth"),
		torch.save(optimizer.state_dict(), OUTPUT_DIR + "unet_opt.pth")],
		freq=100,
	)

	files = glob.glob(f"{DATA_PATH}/*.pth")
	train_files, val_files = files[:-1], files[-1:]
	x_val, y_val = next(data_gen(val_files))

	for i, (x, y) in enumerate(data_gen(train_files)):
		loss, corr = loss_func(model, x, y)

		logger.step ("loss", min(2000, loss))
		logger.step ("corr", corr)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if i % 20 == 0:
			model.eval()
			val_loss, val_corr = loss_func(model, x_val, y_val)
			model.train()
			print(f'val_loss = {val_loss.cpu().data.numpy()} val_corr = {val_corr}')

		if i == 5000: break

