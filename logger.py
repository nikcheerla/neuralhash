
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import random, sys, os, json

import torch

from utils import *


class Logger(object):
	
	def __init__(self, name, features, print_every=20, plot_every=100):

		self.name = name
		self.features = features
		self.print_every, self.plot_every = print_every, plot_every
		self.data = {feature:[] for feature in features}
		self.timestep = {feature:0 for feature in features}
		self.hooks = []

	def add_hook(self, hook):
		self.hooks.append(hook)

	def step(self, feature, x):
		
		if isinstance(x, torch.Tensor):
			x = x.data.cpu().numpy().mean()

		self.data[feature].append(x)
		self.timestep[feature] += 1

		min_timestep = min((t for t in self.timestep.values()))

		if min_timestep % self.print_every == 0 and feature == self.features[-1]:
			print (f"Epoch {min_timestep}: ", end="")
			for feature in self.features:
				print (f"{feature}: {np.mean(self.data[feature][-20:]):0.4f}", end=", ")
			print (f" ... {elapsed():0.2f} sec")

		if min_timestep % self.plot_every == 0 and feature == self.features[-1]:

			for feature in self.features:
				plt.plot(moving_average(np.array(self.data[feature]), n=min(20, self.data[feature]))); 
				plt.savefig(f"{OUTPUT_DIR}/{self.name}_{feature}.jpg");
				plt.cla()

			for hook in self.hooks:
				hook()

		

