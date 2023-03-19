import random
import numpy as np
from skimage import data
from skimage.transform import resize
import torch
from matplotlib import pyplot as plt

im = data.camera()
im = resize(im, (im.shape[0] // 2, im.shape[1] // 2), mode='reflect', preserve_range=True, anti_aliasing=True).astype(np.uint8)
im = torch.from_numpy(im)

plt.imshow(im, cmap='gray')
plt.show()

val = random.randint(0, 255)

out = torch.zeros_like(im)

out[im > val] = 255

out[im <= val] = 0

plt.imshow(out, cmap='gray')
plt.show()
pass
