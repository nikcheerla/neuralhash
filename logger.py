
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import random, sys, os, json, math

import torch
from torchvision import datasets, transforms
import visdom

from utils import *
import IPython


class BaseLogger(object):
    def __init__(self, name, verbose=True):

        self.name = name
        self.data = {}
        self.running_data = {}
        self.reset_running = {}
        self.verbose = verbose
        self.hooks = []

    def add_hook(self, hook, feature="epoch", freq=40):
        self.hooks.append((hook, feature, freq))

    def update(self, feature, x):
        if isinstance(x, torch.Tensor):
            x = x.data.cpu().numpy().mean()

        self.data[feature] = self.data.get(feature, [])
        self.data[feature].append(x)
        if feature not in self.running_data or self.reset_running.pop(feature, False):
            self.running_data[feature] = []
        self.running_data[feature].append(x)

        for hook, hook_feature, freq in self.hooks:
            if feature == hook_feature and len(self.data[feature]) % freq == 0:
                hook(self.data[feature])

    def step(self):
        self.text(f"({self.name}) ", end="")
        for feature in self.running_data.keys():
            if len(self.running_data[feature]) == 0:
                continue
            val = np.mean(self.running_data[feature])
            if float(val).is_integer():
                self.text(f"{feature}: {int(val)}", end=", ")
            else:
                self.text(f"{feature}: {val:0.4f}", end=", ")
            self.reset_running[feature] = True
        self.text(f" ... {elapsed():0.2f} sec")

    def text(self, text, end="\n"):
        raise NotImplementedError()

    def plot(self, data, plot_name, opts={}):
        raise NotImplementedError()

    def images(self, data, image_name):
        raise NotImplementedError()


class Logger(BaseLogger):
    def __init__(self, *args, **kwargs):
        self.results = kwargs.pop("results", "output")
        super().__init__(*args, **kwargs)

    def text(self, text, end="\n"):
        print(text, end=end, flush=True)

    def plot(self, data, plot_name, opts={}):
        np.savez_compressed(f"{self.results}/{plot_name}.npz", data)
        plt.plot(data)
        plt.savefig(f"{self.results}/{plot_name}.jpg")
        plt.clf()


class VisdomLogger(BaseLogger):
    def __init__(self, *args, **kwargs):
        self.port = kwargs.pop("port", 7000)
        self.server = kwargs.pop("server", "35.230.67.129")
        self.env = kwargs.pop("env", "main")
        print(f"Logging to environment {self.env}")
        self.visdom = visdom.Visdom(
            server="http://" + self.server, port=self.port, env=self.env, use_incoming_socket=False
        )
        self.visdom.delete_env(self.env)
        self.windows = {}
        super().__init__(*args, **kwargs)

    def text(self, text, end="\n"):
        print(text, end=end)
        window, old_text = self.windows.get("text", (None, ""))
        if end == "\n":
            end = "<br>"
        display = old_text + text + end

        if window is not None:
            window = self.visdom.text(display, win=window, append=False)
        else:
            window = self.visdom.text(display)

        self.windows["text"] = window, display

    def viz(self, viz_name, method, *args, **kwargs):
        window = self.windows.get(viz_name, None)
        if window is not None:
            window = getattr(self.visdom, method)(*args, **kwargs, win=window)
        else:
            window = getattr(self.visdom, method)(*args, **kwargs, win=window)
        self.windows[viz_name] = window

    def plot(self, data, plot_name, opts={}):

        window = self.windows.get(plot_name, None)
        opts.update({"title": plot_name})

        self.viz(plot_name, "line", np.array(data), opts=opts)

    def images(self, data, image_name, opts={}, resize=64):

        transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(resize), transforms.ToTensor()])
        data = torch.stack([transform(x) for x in data.cpu()])
        data = data.data.cpu().numpy()

        window = self.windows.get(image_name, None)

        opts.update({"title": image_name})
        self.viz(image_name, "images", np.array(data), opts=opts)
