from params import *
import argparse
import os
import numpy as np
import math
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        #self.fc1 = nn.Linear(latent_dim, 1000)
        #self.fc2 = nn.Linear(1000, 7*7*256)
        
        self.Conv2dTrans0 = nn.ConvTranspose2d(latent_dim, 256, 7)
        self.bn0 = nn.BatchNorm2d(256)
        self.Conv2dTrans1 = nn.ConvTranspose2d(256, 128, 8, padding = 0)
        self.bn1 = nn.BatchNorm2d(128)
        self.Conv2dTrans2 = nn.ConvTranspose2d(128, 64, 15, padding = 0)
        self.bn2 = nn.BatchNorm2d(64)
        self.Conv2dTrans3 = nn.ConvTranspose2d(64, 32, 5, padding = 2)
        self.bn3 = nn.BatchNorm2d(32)
        self.Conv2dTrans4 = nn.ConvTranspose2d(32, 1, 5, padding = 2)
    

    def forward(self, z):
        #print("Generator's architecture")
        #out = F.relu(self.fc1(z))
        #out = F.relu(self.fc2(out))
        #print(out.size())
        #out = out.view(out.size(0),256,7,7)
        out = F.relu(self.bn0(self.Conv2dTrans0(z)))
        #print(out.size())
        out = F.relu(self.bn1(self.Conv2dTrans1(out)))
        #print(out.size())
        out = F.relu(self.bn2(self.Conv2dTrans2(out)))
        #print(out.size())
        out = F.relu(self.bn3(self.Conv2dTrans3(out)))
        #print(out.size())
        img = F.tanh(self.Conv2dTrans4(out))
        #print(img.size())
        #img = img.view(img.size(0), *img_shape)
        return img



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 5, padding = 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.mp1 = nn.MaxPool2d(2, stride = 2)
        self.conv2 = nn.Conv2d(64, 128, 5, padding =2)
        self.bn2 = nn.BatchNorm2d(128)
        self.mp2 = nn.MaxPool2d(2, stride = 2)
        self.conv3 = nn.Conv2d(128, 256, 4, padding =0)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 1, padding =0)
        self.bn4 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(4*4*512, 1000)
        self.fc2 = nn.Linear(1000,500)
        self.fc3 = nn.Linear(500,1)
    def forward(self, img):
        #print("Discriminator's architecture")
        #img_flat = img.view(img.size(0), -1)
        out = self.mp1(F.relu(self.bn1(self.conv1(img))))
        #print(out.size())
        out = self.mp2(F.relu(self.bn2(self.conv2(out))))
        #print(out.size())
        out = F.relu(self.bn3(self.conv3(out)))
        #print(out.size())
        out = F.relu(self.bn4(self.conv4(out)))
        #print(out.size())
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        validity = F.sigmoid(out)
        return validity





# Generate some examples
def generate_from_GAN(num_pic, generator, fig_prefix):
  """
  generate class i for
  """
  batch_size = num_pic
  z = Variable(Tensor(np.random.normal(0, 1, (num_pic, latent_dim,1,1)))).float().cuda()
  images = generator(z).cpu().detach().numpy()
  for i in range(batch_size):
    f = plt.figure()
    plt.imshow(np.reshape(images[i], [28,28]))
    f.savefig(fig_prefix + ' {}'.format(i))
  #print(images[0])
