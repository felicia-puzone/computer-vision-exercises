import numpy as np
import torch as torch
import random

n, iC, H, W = input.shape
_, oC, kH, kW = kernel.shape

oH = (H - 1) * s + (kH - 1) + 1
oW = (W - 1) * s + (kW - 1) + 1
out = torch.zeros((n, oC, oH, oW))
# print("out shape:", out.shape)

for dy in range(H):
    for dx in range(W):
        in_t = torch.unsqueeze(input[:, :, dy, dx], 2)
        in_t = torch.unsqueeze(in_t, 3)
        in_t = torch.unsqueeze(in_t, 4)

        k_t = torch.unsqueeze(kernel, 0)

        sum_t = torch.sum(in_t * k_t, dim=1)

        s_dx = dx * s
        s_dy = dy * s
        out[:, :, s_dy:s_dy + kH, s_dx:s_dx + kW] += sum_t
