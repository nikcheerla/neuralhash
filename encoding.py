
from __future__ import print_function

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import random, sys, os, json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from models import DecodingNet
from torchvision import models
from utils import *

from skimage import filters
from skimage.morphology import binary_dilation
import IPython

import transforms

EPSILON = 2e-2
MIN_LOSS = 5e-3
BATCH_SIZE = 80


def encode_binary(image, model, target, max_iter=200, verbose=False):
    
    EPSILON = 2e-2

    image = im.torch(image)
    perturbation_old = None
    # print("Target: ", binary.str(target))
    
    if USE_CUDA: perturbation = nn.Parameter(0.1*torch.randn(image.size()).cuda()+0.0)
    else: perturbation = nn.Parameter(torch.randn(image.size())+0.0)

    # returns an image after a series of transformations
    def p(x):
        x = transforms.resize_rect(x)
        x = transforms.rotate(transforms.scale(x, 0.6, 1.4), max_angle=30)
        x = transforms.gauss(x, min_sigma=0.8, max_sigma=1.2)
        x = transforms.translate(x)
        x = transforms.identity(x)
        return x

    # returns the loss for the image
    def loss_func(model, x):
        predictions = model.forward(x, distribution=p, n=BATCH_SIZE, return_variance=False) # (N, T)
        return F.binary_cross_entropy(predictions, binary.target(target).repeat(BATCH_SIZE, 1)), \
            predictions.mean(dim=0).cpu().data.numpy().round(2)
        # return F.mse_loss(predictions, binary.target(target).repeat(BATCH_SIZE, 1)), \
        #             predictions.mean(dim=0).cpu().data.numpy().round(2)

    def tve_loss(x):
        return ((x[:,:-1,:] - x[:,1:,:])**2).sum() + ((x[:,:,:-1] - x[:,:,1:])**2).sum()
    
    eps = nn.Parameter(torch.tensor(EPSILON).cuda()+0.0)
    alpha, beta = 100, 0.1
    hinge = 0.4

    opt = torch.optim.Adam([perturbation], lr=2e-1)
    losses = []
    preds = []
    p_losses = []
    for i in range(0, max_iter+1):

        def closure():
            opt.zero_grad()

            # perform projected gradient descent by norm bounding the perturbation
            # eps_c = eps.clamp(min=0, max=EPSILON)
            perturbation_zc = (perturbation - perturbation.mean())/(perturbation.std()) * eps
            changed_image = (image + perturbation_zc).clamp(min=0.1, max=0.99)

            loss, predictions = loss_func(model, changed_image)

            perceptual_loss = perturbation_zc.norm(2)
            total = alpha*((loss-hinge).clamp(min=0)) + beta*perceptual_loss
            loss.backward()
            preds.append(predictions)
            losses.append(loss.cpu().data.numpy())
            p_losses.append(perceptual_loss.cpu().data.numpy())
            return loss

        opt.step(closure)
        
        if i % 10 == 0:
            beta *= 1

        if i % 20 == 0:
            # perturbation_zc = perturbation / perturbation.norm(2) * 5
            perturbation_zc = (perturbation - perturbation.mean())/(perturbation.std()) * eps

            changed_image = (image + perturbation_zc).clamp(min=0.1, max=0.99)

            if verbose:
                # print("Epsilon: ", eps.cpu().data.numpy())
                print ("Loss: ", np.mean(losses[-20:]))
                print("Perceptual: ", np.mean(p_losses[-20:]))
                im.save(im.numpy(perturbation), file=f"{OUTPUT_DIR}perturbation.jpg")
                im.save(im.numpy(changed_image), file=f"{OUTPUT_DIR}changed_image.jpg")

                plt.plot(np.array(preds)); 
                plt.savefig(OUTPUT_DIR + "preds.jpg"); plt.cla()
                plt.plot(losses); 
                plt.savefig(OUTPUT_DIR + "loss.jpg"); plt.cla()

                pred = binary.get(np.mean(preds[-20:], axis=0))
                print ("Modified prediction: ", binary.str(pred), binary.distance(pred, target))

        smooth_loss = np.mean(losses[-20:])
        if smooth_loss <= MIN_LOSS:
            break

    print(f"Epsilon: {EPSILON}")
    print ("Loss: ", np.mean(losses[-1]))
    return im.numpy(changed_image)

if __name__ == "__main__":
    target = binary.random(n=TARGET_SIZE)
    model = DecodingNet()
    print("Target: ", binary.str(target))
    encode_binary(im.load("images/car.jpg"), model, target=target, verbose=True)

