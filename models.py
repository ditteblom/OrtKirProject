import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MVCNN_Baseline(nn.Module):
  # from the paper "Multi-view classification with convolutional neural networks" by Seeland and Mäder 2020
  def __init__(self, actfun, alpha):
    super(MVCNN_Baseline, self).__init__()

    self.nView = 2 # default

    if actfun == 'relu':
      act = nn.ReLU()
    elif actfun == 'elu':
      act = nn.ELU(alpha=alpha)
    
    # convolutional layers, CONV_N_M: N is the block and M is the layer in the block
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
                  act,#

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
                  act,#
        )
    # fully connected layer before concatenation
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