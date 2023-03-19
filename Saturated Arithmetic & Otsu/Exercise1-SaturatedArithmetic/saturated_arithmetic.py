import random
import numpy as np
import torch
from skimage import data
from skimage.transform import resize
import cv2
from matplotlib import pyplot as plt

im = data.chelsea()
im = resize(im, (im.shape[0] // 8, im.shape[1] // 8), mode='reflect', preserve_range=True, anti_aliasing=True).astype(np.uint8)
im = np.swapaxes(np.swapaxes(im, 0, 2), 1, 2)

im = torch.from_numpy(im)

a = random.uniform(0,2)
b = random.uniform(-50,50)

im_out = im.to(dtype=torch.float32)

im_out = im_out*a + b

im_out = torch.round(im_out)

im_out[im_out > 255] = 255
im_out[im_out < 0] = 0
im_out =  im_out.numpy().astype(np.uint8)
plt.imshow(np.swapaxes(np.swapaxes(im_out, 0, 2), 1, 0))
plt.show()
pass

