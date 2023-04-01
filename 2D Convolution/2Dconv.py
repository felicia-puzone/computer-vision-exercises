import random
import torch

n = random.randint(2, 6)
iC = random.randint(2, 6)
oC = random.randint(2, 6)
H = random.randint(10, 20)
W = random.randint(10, 20)
kH = random.randint(2, 6)
kW = random.randint(2, 6)

input = torch.rand(n, iC, H, W, dtype=torch.float32)
kernel = torch.rand(oC, iC, kH, kW, dtype=torch.float32)

out = torch.rand(n, oC, H - kH + 1, W - kW + 1 , dtype=torch.float32)

#Prova con l'unsqueeze per evitare il loop su n

for n_im in range(n):
	for i in range(H - kH + 1):
		for j in range(W - kW + 1):
			out[n_im,:,i,j] = torch.sum(kernel*input[n_im,:,i:i+kH,j:j+kW], dim = (1,2,3))
  
  