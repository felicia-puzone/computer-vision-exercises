from io import BytesIO
import numpy as np
import cv2
from skimage import data
from matplotlib import pyplot as plt


"""
	Normal solution for normal people
"""		
im_a = cv2.imread('gallery_0.jpg', cv2.IMREAD_COLOR) 
im_b = cv2.imread('gallery_1.jpg', cv2.IMREAD_COLOR) 
im_a = cv2.cvtColor(im_a, cv2.COLOR_BGR2RGB)
im_b = cv2.cvtColor(im_b, cv2.COLOR_BGR2RGB)

pts_src = np.array([[200,50], [310,100] ,[186, 224], [305,205]])
pts_dst = np.array([[145,60], [330,65] ,[140, 180], [325,190]])

h, status = cv2.findHomography(pts_dst, pts_src)

im_out = cv2.warpPerspective(im_b, h, [im_b.shape[1], im_b.shape[0]])

plt.imshow(im_out)
plt.show()


""" 
Special people solution (to run on aimage platform)
"""

pts_src = np.array([[200,50], [310,100] ,[186, 224], [305,205]])
pts_dst = np.array([[145,60], [330,65] ,[140, 180], [325,190]])

h, status = cv2.findHomography(pts_dst, pts_src)

print('a', im_a.shape)

print('shape1', im_b.shape[1])
print('shape2', im_b.shape[2])

im_out = cv2.warpPerspective(np.swapaxes(np.swapaxes(im_b, 0, 2), 0, 1), h, (im_b.shape[2], im_b.shape[1]))

#im_out = np.swapaxes(im_out, 0, 2)
print('out', im_out.shape)
plt.imshow(im_out)
plt.show()