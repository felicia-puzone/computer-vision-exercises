import random
import numpy as np
from skimage import data
from skimage.transform import resize
import torch
from matplotlib import pyplot as plt

im = data.camera()
im = resize(im, (im.shape[0] // 2, im.shape[1] // 2), mode='reflect', preserve_range=True, anti_aliasing=True).astype(np.uint8)
im = torch.from_numpy(im)

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
    
out = int(torch.argmax(var_b))

