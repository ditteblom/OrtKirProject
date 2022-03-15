# preamble
import glob
import os
from re import X
from PIL import Image
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from skimage.io import imread_collection
import matplotlib.pyplot as plt
import pandas as pd
import os
from unittest import TestLoader
import numpy as np
import glob
import PIL.Image as Image
from tqdm.notebook import tqdm
plt.style.use('seaborn')
import wandb

# train on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training function
def train(model, optimizer, train_loader, test_loader, num_epochs=10, validation = False, scheduler = None):
    def loss_fun(output, target):
        return F.mse_loss(output, target)
    out_dict = {'train_loss': [],
              'test_loss': []}
  
    for epoch in tqdm(range(num_epochs), unit='epoch'):
        model.train()
        #For each epoch
        train_loss = []
        for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, target = data.to(device), target.to(device)
            #Zero the gradients computed for each weight
            optimizer.zero_grad()
            #Forward pass your image through the network
            output = model(data)
            print(output)
            #Compute the loss
            loss = loss_fun(output.float(), target.float())
            #Backward pass through the network
            loss.backward()
            #Update the weights
            optimizer.step()
            train_loss.append(loss.item())

        if scheduler is not None:
          scheduler.step()
          print(f'The current learning rate: {scheduler.get_last_lr()[0]}')

        #Compute the test accuracy
        test_loss = []
        model.eval()

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = model(data)
            test_loss.append(loss_fun(output, target).cpu().item())
    
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['test_loss'].append(np.mean(test_loss))

        # log on wandb.ai
        wandb.log({"train loss": np.mean(train_loss),
                  "test loss": np.mean(test_loss),
                  "learning rate": scheduler.get_last_lr()[0]})

        # to watch on wandb.ai
        wandb.watch(model, log = "all")
        
        #if epoch > 1:
        #  if out_dict['val_acc'][epoch] < out_dict['val_acc'][epoch-1]:
        #    break
        if validation:
          print(f"Loss train: {np.mean(train_loss):.3f}\t validation: {np.mean(test_loss):.3f}\t")
        else:
          print(f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t")#,
              #f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['test_acc'][-1]*100:.1f}%")
    return out_dict
    
# Define a prediction function
def predict(model, testloader):
  ''' 
  Predicts for the test data.
  '''
  predictions = []
  labels = []

  for data, target in testloader:
    data, target = data.to(device), target.to(device)
    with torch.no_grad():
      output = model(data)      

    #Append the predictions and the true labels.
    predictions.append(output.tolist())
    labels.append(target.tolist())

  #Predictions and labels are now a list of list. We want to change that.
  predictions_flat = []
  for prediction in predictions:
    for item in prediction:
      predictions_flat.append(item)

  labels_flat = []
  for label in labels:
    for item in label:
      labels_flat.append(item)

  return np.asarray(labels_flat), np.asarray(predictions_flat)

####
# The following functions is to correct for the score

def score_fluoroscopy(x): # same for all procedures
  if x < 20:
    return 0
  else:
    return -max(-1,(20-x)*(1/10))

def score_time(x):
  if x < 400:
    return 0
  else:
    return -max(-1.0,(400.0-x)*(1/100))

def score_xray(x):
  if x < 40.0:
    return 0
  else:
    return -max(-2.0,(40.0-x)*(0.1))

def score_retries_cannulated_dhs(x):
  if x < 20:
    return 0
  else:
    return max(0,2 - (x-20)*(2/5))

def score_retries_hansson(x):
  if x < 20:
    return 0
  else:
    return -max(-2,(20-x))

# For dynamic hip screw
def drill_dhs(x):
  if x < 0:
    return -max(-5,x)
  elif x < 10:
    return 0
  else:
    -np.max(-5,(10-15)*(5/10))

def guidewire_dist(x):
  if x < -10:
    return -max(-7, (x+10)*3-4)
  elif x < 0:
     return -(x*(2/10)-2)
  elif x < 1:
    return -((x-1)*2)
  elif x > 3:
    return -max(-2,(3-x))
  else:
    return 0

def guidesize_cannulated(x):
  if x < 6:
    return -max(-2,(x-6)*2/2)
  elif x > 10:
    return -((10-x)*(3/10))
  else:
    return 0

def drill_dist_hansson(x):
  if x < 0:
    return -max(-13, x*8/1-5)
  elif x < 3:
    return -((x-3)*5/3)
  elif x > 5:
    return -max(-5, 5-x)
  else:
      return 0

def drill_dist_cannulated(x):
  if x < 0:
    return -max(-5-2, x*3/5-4)
  elif x < 3:
    return -((x-3)*4/3)
  elif x > 5:
    return -max(-4, (5-x)*4/5)
  else:
    return 0

def stepreamer_dist(x):
  if x < 5:
    return -max(-10, (x-5)*10/5)
  elif x > 10:
    return -max(-10, 10-x)
  else:
    return 0


