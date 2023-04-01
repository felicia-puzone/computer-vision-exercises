import random
import torch

n = random.randint(2, 6)
iC = random.randint(2, 6)
H = random.randint(10, 20)
W = random.randint(10, 20)
kH = random.randint(2, 5)
kW = random.randint(2, 5)
s = random.randint(2, 3)
input = torch.rand((n, iC, H, W), dtype=torch.float32)

n, iC, H, W = input.shape
oH = int((H-kH)/s) + 1
oW = int((W - kW)/s) + 1
out = torch.zeros(n, iC, oH, oW)

for i in range(oH):
	for j in range(oW):
		out[:,:,i,j] = torch.amax(input[:,:,i*s:i*s+kH ,j*s:j*s +kW], dim = [2, 3])
  
pass


