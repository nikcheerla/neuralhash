from __future__ import print_function

import numpy as np
import random, sys, os, json

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

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


def getFeatures(model, x):
    x = torch.cat(
        [
            ((x[0] - 0.485) / (0.229)).unsqueeze(0),
            ((x[1] - 0.456) / (0.224)).unsqueeze(0),
            ((x[2] - 0.406) / (0.225)).unsqueeze(0),
        ],
        dim=0,
    )
    x = transforms.identity(x).unsqueeze(0)

    features = []
    prev_feat = x

    for i, module in enumerate(model.features.features._modules.values()):
        next_feat = module(prev_feat)
        features.append(next_feat)
        prev_feat = next_feat

    return features


def gram_matrix(features, normalize=True):
    N, C, H, W = features.shape
    featuresReshaped = features.reshape((N, C, H * W))
    featuresTranspose = featuresReshaped.permute(0, 2, 1)
    ans = torch.matmul(featuresReshaped, featuresTranspose)
    if normalize:
        ans /= H * W * C
    return ans


# features_list is a list of layers of model to calculate content at
# weights_list is a list of corresponding weights for features_list
def content_loss(model, features_list, weights_list, changed_image, original_image):

    original_features = model.getFeatures(original_image)
    changed_features = model.getFeatures(changed_image)

    loss = 0

    for i, layer_number in enumerate(features_list):
        activation_changed = changed_features[layer_number]
        activation_original = original_features[layer_number]

        N, C_1, H_1, W_1 = activation_changed.shape
        F_ij = activation_changed.reshape((C_1, H_1 * W_1))
        P_ij = activation_original.reshape((C_1, H_1 * W_1))
        loss += weights_list[i] * (((F_ij - P_ij).norm(2)) ** 2)

    return loss


def gram_matrix(features, normalize=True):
    N, C, H, W = features.shape
    featuresReshaped = features.reshape((N, C, H * W))
    featuresTranspose = featuresReshaped.permute(0, 2, 1)
    ans = torch.matmul(featuresReshaped, featuresTranspose)
    if normalize:
        ans /= H * W * C
    return ans


# features_list is a list of layers of model to calculate style at
# weights_list is a list of corresponding weights for features_list
def style_loss(model, features_list, weights_list, changed_image, original_image):

    original_features = model.getFeatures(original_image)
    changed_features = model.getFeatures(changed_image)

    loss = 0

    for i, layer_number in enumerate(features_list):
        activation_changed = changed_features[layer_number]
        activation_original = original_features[layer_number]
        changed_gram = gram_matrix(activation_changed)
        original_gram = gram_matrix(activation_original)
        loss += weights_list[i] * (((changed_gram - original_gram).norm(2)) ** 2)

    return loss
