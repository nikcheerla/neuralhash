
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random

from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import IPython


class AdversarialNet(nn.Module):

    def __init__(self, target):
        super(AdversarialNet, self).__init__()
        self.features = models.resnet101(pretrained=True).cuda()
        self.features.eval()
        self.target = target

    def forward(self, x):

        y = x+0.0
        y[0] = (x[0]-0.485)/(0.229)
        y[1] = (x[1]-0.456)/(0.224)
        y[2] = (x[2]-0.406)/(0.225)
        x = y

        def scale(x, min_val=0.8, max_val=1.2):
            C, H, W = 2, 224, 224
            scale_val = random.uniform(min_val, max_val)

            grid = F.affine_grid(torch.eye(3).unsqueeze(0)[:, 0:2], size=torch.Size((1, C, H, W))).cuda()
            img = F.grid_sample(x.unsqueeze(0), grid*scale_val)[0]
            return img

        def rotate(x, max_angle=30):
            C, H, W = 2, 224, 224
            theta = np.radians(random.randint(-max_angle, max_angle))
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c], [0, 0]]).T
            grid = F.affine_grid(torch.FloatTensor(R).unsqueeze(0), size=torch.Size((1, C, H, W))).cuda()
            img = F.grid_sample(x.unsqueeze(0), grid)[0]
            return img

        def distribution(x):
            return rotate(scale(x))

        images = torch.cat([distribution(x).unsqueeze(0) for i in range(0, 10)], dim=0)
        predictions = F.softmax(self.features(images)) [:, self.target].mean()

        return predictions

model = AdversarialNet(398) #target=abacus

tform = transforms.Compose([
            transforms.ToTensor(),
        ])

inverse_tform = transforms.Compose([
            transforms.ToPILImage(),
        ])

def plot(image_data):
    plt.imshow(inverse_tform(image_data.data.cpu()))
    plt.show()

def save(image_data, file):
    plt.imsave(file, inverse_tform(image_data.data.cpu()))
    plt.show()

original = Image.open("cat.jpg")
data = tform(original)

image = Variable(data.cuda())
perturbation = nn.Parameter(torch.randn(image.size()).cuda()+0.0)
opt = optim.Adam([perturbation], lr=0.06)
epsilon, start, fade_in = 5e-3, 4e-2, 1000

losses = []
for i in range(0, 2000):

    epsilon_scaled = (min(i, fade_in) * epsilon + (fade_in - min(i, fade_in))*start)/fade_in
    def closure():
        opt.zero_grad()

        perturbation_zc = (perturbation - perturbation.mean())/(perturbation.std()) * epsilon_scaled
        loss = -model((image + perturbation_zc).clamp(min=0.1, max=0.99))
        loss.backward()
        losses.append(loss.cpu().data.numpy())        
        return loss

    opt.step(closure)

    if i % 10 == 0:
        print ("Loss: ", np.mean(losses[-100:]))
        print ("Epsilon: ", epsilon_scaled)

        perturbation_zc = (perturbation - perturbation.mean())/(perturbation.std()) * epsilon_scaled
        changed_image = (image + perturbation_zc).clamp(min=0.1, max=0.99)

        save(image, file="image.jpg")
        save(perturbation, file="perturbation.jpg")
        save(changed_image, file="changed_image.jpg")

perturbation_zc = (perturbation - perturbation.mean())/(perturbation.std()) * epsilon
changed_image = (image + perturbation_zc).clamp(min=0.1, max=0.99)

print ("Original predictions: ", model(image))
print ("Perturbation: ", model(perturbation_zc))
print ("Modified prediction: ", model(changed_image))


