import torch
import numpy as np

hist = torch.zeros([256])
var_b = torch.zeros([256])

for i in range(256):
    hist[i] = torch.sum(im == i)
    
for i in range(256):
    wb = torch.sum(hist[:i+1])
    wf = torch.sum(hist[i+1:])
    
    if wb == 0 or wf == 0:
        var_b[i] = -1
    else:
        ub = torch.sum(hist[:i+1] * torch.arange(start=0, end=i+1)) / wb
        uf = torch.sum(hist[i+1:] * torch.arange(start=i+1, end=256)) / wf
        var_b[i] = wb * wf * torch.pow(ub - uf, 2)
    
print(var_b)
out = int(torch.argmax(var_b))
