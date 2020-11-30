import torch
import torch.nn as nn
import time
import os
from torch.distributions import Normal
from torchvision import models
import torch.nn.functional as F
import cv2
import numpy as np


class ValueNetwork(torch.nn.Module):
    def __init__(self,n_channels,n_dims):
        super(ValueNetwork, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        self.base_layers = list(self.base_model.children())

        if(n_channels is not 3):
          self.base_layers[0] = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.res = nn.Sequential(*self.base_layers[:-1])
        self.linear0 = self.base_layers[-1]

        self.linear1 = nn.Linear(1000 + n_dims,512)
        self.linear2 = nn.Linear(512,64)
        self.linear3 = nn.Linear(64,1)

    def forward(self, im, x, goal):

      output = self.res(im).flatten(1,-1)
      output = self.linear0(output)
      
      output = torch.cat((output,goal - x),axis = 1)

      output = F.leaky_relu(self.linear1(output))
      output = F.leaky_relu(self.linear2(output))
      output = self.linear3(output)

      return output

class SoftQNetwork(torch.nn.Module):
    def __init__(self, n_channels, action_dims, n_dims):
        super(SoftQNetwork, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        self.base_layers = list(self.base_model.children())

        if(n_channels is not 3):
          self.base_layers[0] = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.res = nn.Sequential(*self.base_layers[:-1])
        self.linear0 = self.base_layers[-1]

        self.linear1 = nn.Linear(1000 + n_dims + action_dims,512)
        self.linear2 = nn.Linear(512,64)
        self.linear3 = nn.Linear(64,1)

    def forward(self, im, x, goal, action):

      output = self.res(im).flatten(1,-1)
      output = self.linear0(output)
      
      output = torch.cat((output,goal - x,action),axis = 1)

      output = F.leaky_relu(self.linear1(output))
      output = F.leaky_relu(self.linear2(output))
      output = self.linear3(output)

      return output


class PolicyNetwork(nn.Module):
    def __init__(self, n_channels, action_dims, n_dims, latent_dims ,device, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.device = device

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.base_model = models.resnet50(pretrained=True)
        self.base_layers = list(self.base_model.children())

        if(n_channels is not 3):
          self.base_layers[0] = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.res = nn.Sequential(*self.base_layers[:-1])
        self.linear0 = self.base_layers[-1]

        self.linear1 = nn.Linear(1000 + n_dims,512)
        self.linear2 = nn.Linear(512,latent_dims)
        
        self.mean_linear = nn.Linear(latent_dims, action_dims)
        
        self.log_std_linear = nn.Linear(latent_dims, action_dims)
        
    def forward(self, im, x, goal):

        output = self.res(im).flatten(1,-1)
        output = self.linear0(output)
        output = torch.cat((output,(goal - x)),axis = 1)
        output = F.leaky_relu(self.linear1(output.float()))
        output = self.linear2(output)
        
        mean    = self.mean_linear(output)
        log_std = self.log_std_linear(output)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, im, x, goal, epsilon=1e-6):
        mean, log_std = self.forward(im, x, goal)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample()
        action = torch.tanh(mean+ std*z.to(self.device))
        log_prob = Normal(mean, std).log_prob(mean+ std*z.to(self.device)) - torch.log(1 - action.pow(2) + epsilon)
        return action, (log_prob[:,0]*log_prob[:,1])[:,None], z, mean, log_std
        
    
    def get_action(self, im, x, goal):
        mean, log_std = self.forward(im,x,goal)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample().to(self.device)
        action = torch.tanh(mean + std*z)
        
        action  = action.cpu()
        return action