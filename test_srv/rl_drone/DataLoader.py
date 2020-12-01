import os
import pdb
import torch
import random
import numpy as np
from config import *

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, ToPILImage, Resize, RandomCrop

import memory_buffer as mem_buf

class CustomDataset(Dataset):
    def __init__(self, buf_len, data_buffer ,transform):

    	"""
			buf_len: Total data points that the buffer can hold
			data_buffer: object of class mem_buf which holds the currect data
			trasnsform: A list of transforms that can be applied on to the images

    	"""

        self.transform = transform
        self.len = buf_len
        self.buffer = data_buffer
        
    def __len__(self):
        return self.len

    def transform(idx):



    	return tensor

    def __getitem__(self, idx):

        image = self.transform(self.buffer.images[idx])
        position = self.buffer.position[idx]

        return (image,position)