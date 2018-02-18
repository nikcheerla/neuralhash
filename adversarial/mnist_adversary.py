from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np
import IPython


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=2)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=2)
        self.fc1 = nn.Linear(1600, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
    	x = x.unsqueeze(1)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x)

model = Net()
model.load_state_dict(torch.load('trained.pth'))
model.cuda()

input_img = Variable(torch.randn(28, 28)).cuda()


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1, shuffle=True, num_workers= 1, pin_memory=True)

data, target = next((x for x in test_loader))

image = Variable(torch.FloatTensor(data[0, 0]).cuda()) 
perturbation = nn.Parameter(torch.randn(28, 28).cuda()+0.0)
opt = optim.Adam([perturbation], lr=0.01)

plt.imshow(image.data.cpu().numpy())
plt.show()

losses = []
norms = []
for i in range(0, 6000):

	def closure():
		opt.zero_grad()
		loss = -model(perturbation.unsqueeze(0) + image.unsqueeze(0))[0, 2]
		norm = torch.norm(perturbation)*0.05
		losses.append(loss.cpu().data.numpy())
		norms.append(norm.cpu().data.numpy())
		loss = loss + norm
		loss.backward()
		return loss

	opt.step(closure)

	if i % 50 == 0:
		print ("Loss: ", np.mean(losses[-100:]))
		print ("Norms : ", np.mean(norms[-100:]))

changed_image = image + perturbation
print ("Original predictions: ", model(image.unsqueeze(0))[0, 2])
print ("Perturbation: ", model(perturbation.unsqueeze(0))[0, 2])
print ("Modified prediction: ", model(changed_image.unsqueeze(0))[0, 2])

plt.imshow(image.data.cpu().numpy()); plt.show()
plt.imshow(perturbation.data.cpu().numpy()); plt.show()
plt.imshow(changed_image.data.cpu().numpy()); plt.show()
