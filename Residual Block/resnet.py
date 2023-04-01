import torch.nn as nn
import torch


class ResidualBlock(nn.Module):
    def __init__(self, inplanes=3, planes= 3, stride = 1):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, padding = 1, bias = False)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding = 1, bias = False)
        self.convG = nn.Conv2d(inplanes, planes, 1, stride)
        
        self.batch1 = nn.BatchNorm2d(planes)
        self.batch2 = nn.BatchNorm2d(planes)
        self.batchG = nn.BatchNorm2d(planes)
        
        self.relu = nn.ReLU()
        

        
    def forward(self, x):
        if self.inplanes == self.planes and self.stride == 1:
            G_x = x
        else:
            G_x = self.convG(x)
            G_x = self.relu(G_x)
            G_x = self.batchG(G_x)
            
        F_x = self.conv1(x)
        F_x = self.batch1(F_x)
        F_x = self.relu(F_x)
        
        F_x = self.conv2(F_x)
        F_x = self.batch2(F_x)
        
        
        out =  F_x + G_x
        out = self.relu(out)
        return out
    

if __name__ == "__main__":
    x = torch.rand(3, 100, 100, dtype=torch.float32)
    x = x.unsqueeze(0)
    
    net = ResidualBlock(stride=3)
    
    out = net(x)
    print(out)
    
    pass
        