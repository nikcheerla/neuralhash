
import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.randn(1, 3, 32, 32).requires_grad_().float()
b = torch.tensor(0.01).requires_grad_().float()

def f(a, b):
	grid = F.affine_grid(torch.eye(3).unsqueeze(0)[:, 0:2], \
				size=torch.Size((1, 2, 64, 64)))
	return F.grid_sample(a+b, grid).mean()

f_val = f(a, b)

(dfda,) = torch.autograd.grad(f_val, [a], create_graph=True, retain_graph=True)
print (f"df/da={dfda.shape}, requires_grad={dfda.requires_grad}")

def g(a, b):
	grid = F.affine_grid(torch.eye(3).unsqueeze(0)[:, 0:2], \
				size=torch.Size((1, 2, 64, 64)))

	# Basically the same function, just with multiplication instead of addition
	return F.grid_sample(a*b, grid).mean() 

g_val = g(a, b)

(dgda,) = torch.autograd.grad(g_val, [a], create_graph=True, retain_graph=True)
print (f"dg/da={dgda.shape}, requires_grad={dgda.requires_grad}")
