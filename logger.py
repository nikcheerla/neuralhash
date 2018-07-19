
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import random, sys, os, json

import torch

from utils import *


class Logger(object):
    
    def __init__(self, name, features, print_every=20, plot_every=100, verbose=True):

        self.name = name
        self.features = features
        self.print_every, self.plot_every = print_every, plot_every
        self.data = {feature:[] for feature in features}
        self.timestep = {feature:0 for feature in features}
        self.verbose = verbose
        self.hooks = []

    def add_hook(self, hook, freq=40):
        self.hooks.append((hook, freq))

    def step(self, feature, x):
        
        if isinstance(x, torch.Tensor):
            x = x.data.cpu().numpy().mean()

        self.data[feature].append(x)
        self.timestep[feature] += 1

        min_timestep = min((t for t in self.timestep.values()))

        if min_timestep % self.print_every == 0 and feature == self.features[-1] and self.verbose:
            print (f"({self.name}) Epoch {min_timestep}: ", end="")
            for feature in self.features:
                print (f"{feature}: {np.mean(self.data[feature][-self.print_every:]):0.4f}", end=", ")
            print (f" ... {elapsed():0.2f} sec", flush=True)

        if min_timestep % self.plot_every == 0 and feature == self.features[-1] and self.verbose:

            for feature in self.features:
                self.plot(np.array(self.data[feature]), \
                    f"{OUTPUT_DIR}{self.name}_{feature}.jpg")

        for hook, freq in self.hooks:
            if min_timestep % freq == 0 and feature == self.features[-1] and self.verbose:
                hook()

    def plot(self, data, plot_file):

        np.savez_compressed(plot_file[:-4] + ".npz", data)
        plt.plot(data)
        plt.savefig(plot_file); 
        plt.clf()


