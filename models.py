import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision

# Define models
class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.convolutional3d = nn.Sequential(
                  # CONV 3D_1
                  nn.Conv3d(in_channels=3, out_channels=8, kernel_size=3, stride = 1, padding= 1),
                  nn.ReLU(),

                  # CONV 3D_2
                  nn.Conv3d(in_channels=8, out_channels=16, kernel_size=2, stride = 1, padding= 0),
                  nn.ReLU(),
        )
        self.convolutional = nn.Sequential(
                  # CONV 1_1
                  nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),

                  # CONV 1_2
                  nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                  nn.MaxPool2d(kernel_size=2),
                  nn.ReLU(),

                  # CONV 2_1
                  nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),

                  # CONV 2_2
                  nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
        )
        self.fully_connected = nn.Sequential(
                nn.Linear(64*64*64, 1),  
                )

    def forward(self, x):
      x = self.convolutional3d(x)
      x = torch.squeeze(x,dim = 2)
      x = self.convolutional(x)
      x = torch.flatten(x,1)
      x = self.fully_connected(x).squeeze()
      return x

# VGG16 takes an image of 224 x 224 with 3 channels as input
class VGG16(nn.Module):
  def __init__(self,):
    super(VGG16, self).__init__()

    self.convolutional3d_2_2d = nn.Sequential(
                  # CONV 3D_1
                  nn.Conv3d(in_channels=3, out_channels=8, kernel_size=3, stride = 1, padding= 1),
                  nn.ReLU(inplace=True),

                  # CONV 3D_2
                  nn.Conv3d(in_channels=8, out_channels=3, kernel_size=2, stride = 1, padding= 0),
                  nn.ReLU(inplace=True),
        )

    self.vgg16 = torchvision.models.vgg16(pretrained=True)

    #Freeze the parameters (by using this method, you can freeze the parameters in the convolutional layers and/or the fully connected layers)
    for param in self.vgg16.parameters(): # convolutional
      param.requires_grad = False

    #for param in self.vgg16.classifier.parameters(): # fully connected
    #  param.requires_grad = False

    #Modify the last layer
    n_features = self.vgg16.classifier[3].in_features #Number of features (inputs) in the last layer we want to keep
    features = list(self.vgg16.classifier.children())[:-4] # remove the last two FC layers
    features.extend([nn.Linear(n_features, 1)]) #1 output (numeric)
    self.vgg16.classifier = nn.Sequential(*features)
    
  def forward(self, x):
    x = self.convolutional3d_2_2d(x)
    x = self.vgg16(x.squeeze())
    return x.squeeze()

# Inception v-3 takes an image of 299 x 299 with 3 channels as input
class Inception_v3(nn.Module):
  def __init__(self,):
    super(Inception_v3, self).__init__()

    self.convolutional3d_2_2d = nn.Sequential(
                  # CONV 3D_1
                  nn.Conv3d(in_channels=3, out_channels=8, kernel_size=3, stride = 1, padding= 1),
                  nn.ReLU(inplace=True),

                  # CONV 3D_2
                  nn.Conv3d(in_channels=8, out_channels=3, kernel_size=2, stride = 1, padding= 0),
                  nn.ReLU(inplace=True),
        )

    self.inception = torchvision.models.inception_v3(pretrained=True)

    #Freeze the parameters
    for param in self.inception.parameters():
      param.require_grad = False

    #Remove the last layer
    n_features = self.inception.fc.in_features #Number of features (inputs) in the last layer
    self.inception.fc = nn.Linear(n_features,1)
    
  def forward(self, x):
    x = self.convolutional3d_2_2d(x)
    x = self.inception(x.squeeze())
    return x[0]