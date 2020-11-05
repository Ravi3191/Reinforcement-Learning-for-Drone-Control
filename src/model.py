import torch
import torch.nn as nn
import time
import os
from torchvision import models
import cv2
import numpy as np

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        self.base_model = models.resnet50(pretrained=True)
        self.base_layers = list(self.base_model.children()) 

    def forward(self, input):


        return out

def unit_test():
    num_minibatch = 2
    num_channels = 
    rgb = torch.randn(num_minibatch, num_channels, 64, 2048)
    rtf_net = Net()
    
    output=rtf_net(rgb,rgb,rgb)
    print(output.shape)
    

if __name__ == '__main__':
    unit_test()
