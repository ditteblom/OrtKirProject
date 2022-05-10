import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchvision import transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

# Define models
class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
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
                nn.Linear(64**3, 1),  
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
                  nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, stride = 1, padding= 1),
                  nn.ReLU(inplace=True),

                  # CONV 3D_2
                  nn.Conv3d(in_channels=8, out_channels=3, kernel_size=2, stride = 1, padding= 0),
                  nn.ReLU(inplace=True),
        )

    self.inception = torchvision.models.inception_v3(pretrained=True)

    #Freeze the parameters
    for param in self.inception.parameters():
      param.requires_grad = False

    #Remove the last layer
    n_features = self.inception.fc.in_features #Number of features (inputs) in the last layer
    self.inception.fc = nn.Linear(n_features,1)
    
  def forward(self, x):
    x = self.convolutional3d_2_2d(x)
    x = self.inception(x.squeesze())
    return x[0]

class UNet(nn.Module):
  def __init__(self, init_size = 200, init_layers=64, dropout_p = 0.2):
      super(UNet, self).__init__()
      self.init_size = init_size
      init_layers = np.log2(init_layers).astype(int)
      # encoder (downsampling)
      self.encoder_0 = nn.Sequential(
                        nn.Conv2d(in_channels=3, out_channels=2**(init_layers), kernel_size=3),
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

  def forward(self, x_in):
    x_in = x_in.permute(2,0,1,3,4)
    self.nView = x_in.shape[0]
    outputs = []

    # forward pass each view
    for i in range(self.nView):
      x = x_in[i]
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
      x = self.decoder_0(torch.cat([transforms.CenterCrop(int((self.init_size-8)/2-10))(e1), x], 1))
      x = self.upscale_1(x)
      x = self.decoder_1(torch.cat([transforms.CenterCrop(self.init_size-36)(e0), x], 1))
      x = self.fully_connected(x)
      outputs += [x]

    # concatenate output from different views
    out = torch.stack(outputs, dim=1)

    return out

class MVCNN_VGG19(nn.Module):
  # from the paper "Multi-view classification with convolutional neural networks" by Seeland and Mäder 2020
  # Late
  def __init__(self):
    super(MVCNN_VGG19, self).__init__()

    self.nView = 2 # default
    self.vgg19 = torchvision.models.vgg19(pretrained=True).to(device)

    #Freeze the parameters
    for param in self.vgg19.parameters():
      param.requires_grad = False

    #Modify the last layer
    n_features = self.vgg19.classifier[3].in_features #Number of features (inputs) in the last layer we want to keep
    features = list(self.vgg19.classifier.children())[:-4] # remove the last two FC layers
    features.extend([nn.Linear(n_features, 1024), nn.ReLU()])
    self.vgg19.classifier = nn.Sequential(*features)

    # fully connected layer after concatenation
    self.linear = nn.Sequential(
      nn.Linear(1024*self.nView, 1)
    )
    
  def forward(self, x):
    x = x.permute(2,0,1,3,4)
    self.nView = x.shape[0]
    N1_out = []

    # forward pass each view
    for i in range(self.nView):
      N1_out += [self.vgg19(x[i])]

    # concatenate output from different views
    x = torch.cat(N1_out, dim = 1)
    x = self.linear(x)
    return x

class MVCNN_VGG19_early(nn.Module):
  # from the paper "Multi-view classification with convolutional neural networks" by Seeland and Mäder 2020
  # early
  def __init__(self):
    super(MVCNN_VGG19_early, self).__init__()

    self.nView = 2 # default
    self.vgg19 = torchvision.models.vgg19(pretrained=True).to(device)

    #Freeze the parameters
    for param in self.vgg19.parameters():
      param.requires_grad = False

    #Remove the classifier the last layer
    n_features = self.vgg19.classifier[0].in_features #Number of features (inputs) in the last layer we want to keep
    self.vgg19.classifier = Identity()

    # fc layer for single view
    self.fc = nn.Sequential(
      nn.Linear(n_features, 1024), nn.ReLU(),
    )

    # fully connected layer after concatenation
    self.linear = nn.Sequential(
      nn.Linear(1024*self.nView, 1)
    )
    
  def forward(self, x):
    x = x.permute(2,0,1,3,4)
    self.nView = x.shape[0]
    N1_out = []

    # forward pass each view
    for i in range(self.nView):
      tmp = self.vgg19(x[i])
      N1_out += [self.fc(tmp)]

    # concatenate output from different views
    x = torch.cat(N1_out, dim = 1)
    x = self.linear(x)
    return x

class MVCNN_Inception(nn.Module):
  # from the paper "Multi-view classification with convolutional neural networks" by Seeland and Mäder 2020
  # Late 
  def __init__(self):
    super(MVCNN_Inception, self).__init__()

    self.nView = 2 # default
    self.inception = torchvision.models.inception_v3(pretrained=True)

    #Freeze the parameters
    for param in self.inception.parameters():
      param.requires_grad = False

    #Modify the last layer
    n_features = self.inception.fc.in_features #Number of features (inputs) in the last layer
    self.inception.fc = nn.Linear(n_features,1024)

    # fully connected layer after concatenation
    self.linear = nn.Sequential(nn.ReLU(),
      nn.Linear(1024*self.nView, 1)
    )
    
  def forward(self, x):
    x = x.permute(2,0,1,3,4)
    self.nView = x.shape[0]
    N1_out = []

    # forward pass each view
    for i in range(self.nView):
      N1_out += [self.inception(x[i])]

    # concatenate output from different views
    x = torch.cat(N1_out, dim = 1)
    x = self.linear(x)
    return x

class MVCNN_Baseline(nn.Module):
  # from the paper "Multi-view classification with convolutional neural networks" by Seeland and Mäder 2020
  def __init__(self, actfun, alpha):
    super(MVCNN_Baseline, self).__init__()

    self.nView = 2 # default

    if actfun == 'relu':
      act = nn.ReLU()
    elif actfun == 'elu':
      act = nn.ELU(alpha=alpha)
    
    # convolutional layers
    self.convolutional = nn.Sequential(
                  # CONV 1_1
                  nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=1, padding=3),
                  act,

                  # CONV 1_2
                  nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=1, padding=3),
                  act,

                  # CONV 1_3
                  nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),

                  # MAXPOOL 1, 224 -> 112
                  nn.MaxPool2d(kernel_size=2),
                  act,

                  # CONV 2_1
                  nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
                  act,

                  # CONV 2_2
                  nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
                  act,

                  # CONV 2_3
                  nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),

                  # MAXPOOL 2, 112 -> 56
                  nn.MaxPool2d(kernel_size=2),
                  act,#act,

                  # CONV 3_1
                  nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
                  act,

                  # CONV 3_2
                  nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                  act,

                  # CONV 3_3
                  nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),

                  # MAXPOOL 3, 56 -> 28
                  nn.MaxPool2d(kernel_size=2),
                  act,#act
        )

    self.fc = nn.Sequential(
      nn.Linear(28**2*32,1024), act
    )

    # fully connected layer after concatenation
    self.linear = nn.Sequential(
      nn.Linear(1024*self.nView, 1)
    )
    
  def forward(self, x):
    x = x.permute(2,0,1,3,4)
    self.nView = x.shape[0]
    N1_out = []

    # forward pass each view
    for i in range(self.nView):
      temp = self.convolutional(x[i])
      temp = torch.flatten(temp,1)
      N1_out.append(self.fc(temp))   
    # concatenate output from different views
    x = torch.cat(N1_out, dim = 1)
    x = self.linear(x)
    return x

class MVCNN_UNet(nn.Module):
  # from the paper "Multi-view classification with convolutional neural networks" by Seeland and Mäder 2020
  # Late 
  def __init__(self):
    super(MVCNN_UNet, self).__init__()

    self.nView = 2 # default
    self.unet = UNet()

    # postnet after UNet to get features, size of image is init_size-36
    self.encoderConv = nn.Sequential(
                    nn.Conv2d(1,
                             64,
                             kernel_size=5, stride=1,
                             padding=0),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64,
                             64,
                             kernel_size=5, stride=1,
                             padding=0),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64,
                             64,
                             kernel_size=5, stride=1,
                             padding=0),
                    nn.BatchNorm2d(64),
                    nn.ReLU())

    # init_size -40-3*4 = init_size - 52
    # if init_size is 128 then now image is 76 with 64 channels
    self.encoderFC = nn.Sequential(
                        nn.Linear(76**2*64,2048),nn.ReLU(), 
                        nn.Linear(2048,1024), nn.ReLU())


    # fully connected layer after concatenation
    self.linear = nn.Sequential(
      nn.Linear(1024*self.nView, 1)
    )
    
  def forward(self, x):
    x = x.permute(2,0,1,3,4)
    self.nView = x.shape[0]
    segs_out = []
    N1_out = []

    # forward pass each view
    for i in range(self.nView):
      tmp_seg = self.unet(x[i])
      segs_out += [tmp_seg]
      tmp = self.encoderConv(tmp_seg)
      tmp = torch.flatten(tmp,1)
      N1_out += [self.encoderFC(tmp)]  

    segs_out = torch.stack(segs_out)
    segs_out = segs_out.permute(1,0,2,3,4)
    # concatenate output from different views
    x = torch.cat(N1_out, dim = 1)
    x = self.linear(x)
    return segs_out, x