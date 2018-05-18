from __future__ import print_function

import numpy as np
import random, sys, os, json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchvision import models
from utils import *

from skimage import filters
from skimage.morphology import binary_dilation

import IPython

import transforms

"""Decoding network that tries to predict a
binary value of size target_size """
class DecodingNet(nn.Module):

    def __init__(self):
        super(DecodingNet, self).__init__()

        self.features = models.vgg11(pretrained=True)
        self.features.classifier = nn.Sequential(
            nn.Linear(25088, TARGET_SIZE))

        self.features.eval()

        if USE_CUDA: self.cuda()

    def forward(self, x, verbose=False, distribution=transforms.identity, 
                    n=1, return_variance=False):

        # make sure to center the image and divide by standard deviation
        x = torch.cat([((x[0]-0.485)/(0.229)).unsqueeze(0),
            ((x[1]-0.456)/(0.224)).unsqueeze(0),
            ((x[2]-0.406)/(0.225)).unsqueeze(0)], dim=0)

        images = torch.cat([distribution(x).unsqueeze(0) for i in range(0, n)], dim=0)
        predictions = (self.features(images)) + 0.5
        if return_variance:
            return predictions.mean(dim=0), predictions.std(dim=0)
        return predictions

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)






"""More complex network that tries to predict a
binary value of size target_size """
class DecodingDNN(nn.Module):

    def __init__(self):
        super(DecodingDNN, self).__init__()

        self.features = models.resnet101(pretrained=True)
        self.classifier = nn.Linear(1024*2*2, TARGET_SIZE*2)
        
        mask = Variable(torch.bernoulli(torch.ones(TARGET_SIZE*2, 1024*2*2)*0.03))/0.03
        self.classifier.weight.data = self.classifier.weight.data*mask.data
        self.features.eval()

        if USE_CUDA: self.cuda()

    def forward(self, x, verbose=False, distribution=transforms.identity, 
                    n=1, return_variance=False):

        # make sure to center the image and divide by standard deviation
        x = torch.cat([((x[0]-0.485)/(0.229)).unsqueeze(0),
            ((x[1]-0.456)/(0.224)).unsqueeze(0),
            ((x[2]-0.406)/(0.225)).unsqueeze(0)], dim=0)

        x = torch.cat([distribution(x).unsqueeze(0) for i in range(0, n)], dim=0)
        for layer in list(self.features._modules.values())[0:6]:
            x = layer(x)

        module = list(self.features._modules.values())[6]
        for layer in list(module._modules.values())[0:5]:
            x = layer(x)

        x = F.max_pool2d(x, x.size(2)//2) + F.avg_pool2d(x, x.size(2)//2)
        x = x.view(x.size(0), -1)
        x = (x - x.mean(dim=1, keepdim=True))/(x.std(dim=1, keepdim=True))
        x = self.classifier(x).view(x.size(0), TARGET_SIZE, 2)
        x = x.mean(dim=0)
        predictions = F.log_softmax(x)[:, 0]

        return predictions

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)


if __name__ == "__main__":

    model = DecodingDNN()
    model.forward(Variable(torch.randn(3, 224, 224)))
