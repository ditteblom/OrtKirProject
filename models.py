import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchvision import transforms

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
    x = self.inception(x.squeesze())
    return x[0]

class UNet(nn.Module):
  def __init__(self, init_size = 256, init_layers=64, dropout_p = 0.2):
      super().__init__()
      self.init_size = init_size
      init_layers = np.log2(init_layers).astype(int)
      # encoder (downsampling)
      self.encoder_0 = nn.Sequential(
                        nn.Conv2d(in_channels=1, out_channels=2**(init_layers), kernel_size=3),
                        nn.ReLU(),
                        nn.Dropout(p = dropout_p),
                        nn.BatchNorm2d(num_features = 2**(init_layers)),
                        nn.Conv2d(in_channels=2**(init_layers), out_channels=2**(init_layers), kernel_size=3),
                        nn.ReLU(),
                        nn.Dropout(p = dropout_p),
                        nn.BatchNorm2d(num_features = 2**(init_layers))
                      )
      
      self.pool_0 = nn.MaxPool2d(2,2)
      
      self.encoder_1 = nn.Sequential(
                        nn.Conv2d(in_channels=2**(init_layers), out_channels=2**(init_layers+1), kernel_size=3),
                        nn.ReLU(),
                        nn.Dropout(p = dropout_p),
                        nn.BatchNorm2d(num_features = 2**(init_layers+1)),
                        nn.Conv2d(in_channels=2**(init_layers+1), out_channels=2**(init_layers+1), kernel_size=3),
                        nn.ReLU(),
                        nn.Dropout(p = dropout_p),
                        nn.BatchNorm2d(num_features = 2**(init_layers+1))
                      )
      
      self.pool_1 = nn.MaxPool2d(2,2)

      self.bottleneck = nn.Sequential(
                          nn.Conv2d(in_channels=2**(init_layers+1), out_channels=2**(init_layers+2), kernel_size=3),
                          nn.ReLU(),
                          nn.Dropout(p = dropout_p),
                          nn.BatchNorm2d(num_features = 2**(init_layers+2)),
                          nn.Conv2d(in_channels=2**(init_layers+2), out_channels=2**(init_layers+2), kernel_size=3),
                          nn.ReLU(),
                          nn.Dropout(p = dropout_p),
                          nn.BatchNorm2d(num_features = 2**(init_layers+2))
                      )
      
      self.upscale_0 = nn.ConvTranspose2d(in_channels=2**(init_layers+2), out_channels=2**(init_layers+1), kernel_size=2, stride=2)

      self.decoder_0 = nn.Sequential(
                          nn.Conv2d(in_channels=2**(init_layers+2), out_channels=2**(init_layers+1), kernel_size=3),
                          nn.ReLU(),
                          nn.Dropout(p = dropout_p),
                          nn.BatchNorm2d(num_features = 2**(init_layers+1)),
                          nn.Conv2d(in_channels=2**(init_layers+1), out_channels=2**(init_layers+1), kernel_size=3),
                          nn.ReLU(),
                          nn.Dropout(p = dropout_p),
                          nn.BatchNorm2d(num_features = 2**(init_layers+1))
                      )
        
      self.upscale_1 = nn.ConvTranspose2d(in_channels=2**(init_layers+1), out_channels=2**(init_layers), kernel_size=2, stride=2)

      self.decoder_1 = nn.Sequential(
                          nn.Conv2d(in_channels=2**(init_layers+1), out_channels=2**(init_layers), kernel_size=3),
                          nn.ReLU(),
                          nn.Conv2d(in_channels=2**(init_layers), out_channels=2**(init_layers), kernel_size=3),
                          nn.ReLU(),
                      )
      
      self.fully_connected = nn.Conv2d(in_channels=2**(init_layers), out_channels=2, kernel_size=1)

  def forward(self, x):
      # Encoder
      x = self.encoder_0(x)
      e0 = x #save e0 for the skip-connection.

      x = self.pool_0(x)
      x = self.encoder_1(x)
      e1 = x #save e1 for the skip-connection.
      x = self.pool_1(x)

      #Bottleneck
      x = self.bottleneck(x)

      #Decoder 
      x = self.upscale_0(x)
      x = self.decoder_0(torch.cat([transforms.CenterCrop(int((self.init_size-4)/2-8))(e1), x], 1))
      x = self.upscale_1(x)
      x = self.decoder_1(torch.cat([transforms.CenterCrop(self.init_size-36)(e0), x], 1))
      x = self.fully_connected(x)
      
      return x