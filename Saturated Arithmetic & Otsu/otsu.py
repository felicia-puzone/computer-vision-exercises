import random
import numpy as np
from skimage import data
from skimage.transform import resize
import torch
from matplotlib import pyplot as plt

im = data.camera()
im = resize(im, (im.shape[0] // 2, im.shape[1] // 2), mode='reflect', preserve_range=True, anti_aliasing=True).astype(np.uint8)
im = torch.from_numpy(im)

# variance_b = Wb*Wf (ub-uf)^2
# iterate thresholds and compute variance_b[t]
# the correct threshold is the one corresponding to higher variance_b

hist = torch.zeros([256])
indexes = torch.zeros([256])
var_b = torch.zeros([256])

for i in range(256):
    hist[i] = torch.sum(im == i)
    indexes[i] = i

#plt.bar(np.arange(0,255, 1, dtype=int), hist)
#plt.show()
    
for i in range(256):
    wb = torch.sum(hist[:i+1] / torch.sum(hist))
    wf = torch.sum(hist[i+1:] / torch.sum(hist))
    
    ub = torch.sum(hist[:i+1] * indexes[:i+1])/ torch.sum(hist[:i+1])
    uf = torch.sum(hist[i+1:] * indexes[i+1:])/ torch.sum(hist[i+1:])
    
    var_b[i] = wb*wf*((ub -uf)**2)

var_b[torch.isnan(var_b)] = 0
    
thresh = int(torch.argmax(var_b)) 

out = torch.zeros_like(im)

out[im > thresh] = 255
out[im <= thresh] = 0

print(thresh)

plt.imshow(out, cmap='gray')
plt.show()

pass

