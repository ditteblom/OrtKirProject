# preamble
from re import X
from attr import dataclass
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
plt.style.use('seaborn')
import wandb
import pandas as pd
from models import UNet
import torch.nn as nn
from datetime import datetime

# train on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training function
def train(model, optimizer, loss_fun, train_loader, test_loader, num_epochs=10, validation = False, scheduler = None, seg = False):
    out_dict = {'train_loss': [],
              'test_loss': []}
  
    for epoch in range(num_epochs):
        model.train()
        #For each epoch
        train_loss = []
        for minibatch_no, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            if seg:
              #Center crop the segmentation
              target = transforms.CenterCrop(216)(target).long()

            #Zero the gradients computed for each weight
            optimizer.zero_grad()
            #Forward pass your image through the network
            output = model(data)
            #Compute the loss
            if seg:
              loss = loss_fun(output, target)
            else:
              loss = loss_fun(output.float(), target.unsqueeze(1).float())
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
            if seg:
              #Center crop the segmentation
              target = transforms.CenterCrop()(target).long()
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
    return -max(-5,(10-x)*(5/10))

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

def segmentation_results(seg, gt):
  '''
  Returns a vector of dice scores for each image in the batch
  Input: 
    seg, our segmentation - tensor of size nbatch x 1 x nrows x npixels
    gt,  the ground truth - tensor of size nbatch x 1 x 

  Implemented from: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
  '''
  assert seg.shape == gt.shape, 'Segmentation and ground truth should have the same shape.'

  #Set machine epsilon such that we don't divide with zero.
  eps = 2e-16

  nbatch, _, _, _ = seg.shape
  
  dices      = torch.zeros(nbatch)
  IoUs       = torch.zeros(nbatch)

  accuracies    = torch.zeros(nbatch)
  sensitivities = torch.zeros(nbatch)
  specificities = torch.zeros(nbatch)

  for i in range(nbatch):
    seg_i, gt_i = seg[i].squeeze(0), gt[i].squeeze(0)

    #Find the IoU for two segmentations
    overlap = torch.logical_and(seg_i.bool(), gt_i.bool()).sum().item()
    union   = torch.logical_or(seg_i.bool(), gt_i.bool()).sum().item()
    
    
    IoU = overlap/(union + eps)
    IoUs[i] = IoU

    #Use confusion matrix to get tn, fp, fn and tp for the i'th image.
    tn, fp, fn, tp = confusion_matrix(gt_i.reshape(-1,), seg_i.reshape(-1,), labels=[0,1]).ravel()

    #Dice score: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    dices[i] = (2*tp)/(2*tp + fp + fn + eps)

    #Check the table here: https://en.wikipedia.org/wiki/Sensitivity_and_specificity
    accuracies[i] = (tp + tn)/(tp + tn + fp + fn + eps)
    sensitivities[i] = (tp)/(tp+fn + eps)
    specificities[i] = (tn)/(tn+fp + eps)

  return dices, IoUs, accuracies, sensitivities, specificities  

def evaluate_split(model, dataloader, loss_fn):
  '''
  Evaluates all the metrics for a given dataloader.
  '''

  avg_loss = 0
  avg_dice = 0
  avg_IoU  = 0
  avg_acc  = 0
  avg_sens = 0
  avg_spec = 0

  
  with torch.no_grad():
    for k, data in enumerate(dataloader):
      #Fetch the input data
      image, score, gray, an = data

      _, _, _, sizex, sizey = image.shape
      
      #Center crop the segmentation
      an = transforms.CenterCrop((sizex-40,sizey-40))(an).long()
      image, an = image.to(device), an.to(device)

      #Forward pass
      pred = model(image)

      #Calculate the loss
      loss = loss_fn(pred, an.squeeze(1))

      #Update the different evaluation metrics
      avg_loss += loss.item() / len(dataloader)

      #Get the predicted segmentation
      pred_seg = torch.softmax(pred, dim=1).argmax(axis=1).unsqueeze(1)
      

      dices, IoUs, accuracies, sensitivities, specificities = segmentation_results(pred_seg.cpu(), an.cpu())

      avg_dice += dices.mean().item() / len(dataloader)
      avg_IoU  += IoUs.mean().item() / len(dataloader)
      avg_acc  += accuracies.mean().item() / len(dataloader)
      avg_sens += sensitivities.mean().item() / len(dataloader)
      avg_spec += specificities.mean().item() / len(dataloader)

  return avg_loss, avg_dice, avg_IoU, avg_acc, avg_sens, avg_spec

def train_anno(train_loader, val_loader=None, test_loader=None, epochs = 100, plot=False, return_results=False):
  '''
  Train function!
  '''
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  NetScrew = UNet()
  NetScrew.to(device)
  run_name = 'UNet_' + datetime.now().strftime('_%y%B%d_%H%M_%S')

  print(run_name)

  optimizerScrew = torch.optim.Adam(NetScrew.parameters(), lr = 0.001, betas=(0.99,0.999)) # beta1 = momentum for Adam, use 0.99 like in UNet paper
  #decayRate = 0.96
  #schedulerBone = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizerBone, gamma=decayRate)
  #schedulerScrew = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizerScrew, gamma=decayRate)

  loss_fn = nn.CrossEntropyLoss()

  #Make a dataframe containing the results from the training.
  if return_results:
    columns = ['epoch', 'avg_train_loss', 'avg_train_dice', 'avg_train_IoU', 'avg_train_acc', 'avg_train_sens', 'avg_train_spec']
    if val_loader is not None:
      columns += ['avg_val_loss', 'avg_val_dice', 'avg_val_IoU', 'avg_val_acc', 'avg_val_sens', 'avg_val_spec']
    if test_loader is not None:
      columns += ['avg_test_loss', 'avg_test_dice', 'avg_test_IoU', 'avg_test_acc', 'avg_test_sens', 'avg_test_spec']

    #Construct the pd.DataFrame with the results for each epoch.
    results = pd.DataFrame(columns = columns)
    
  for epoch in range(epochs):

    #Change the epoch index.
    epoch += 1

    #Make a dictionary for returning the results in a dataframe.
    if return_results:
      epoch_results = {}

    #Training part of the epoch
    NetScrew.train()

    avg_train_lossScrew = 0
    avg_train_diceScrew = 0
    avg_train_IoUScrew  = 0
    avg_train_accScrew  = 0
    avg_train_sensScrew = 0
    avg_train_specScrew = 0


    for k, data in enumerate(train_loader):
      # fetch data
      image, _, _, an = data
      _, _, _, rows, _ = image.shape
      #Center crop the segmentation
      an = transforms.CenterCrop(rows-40)(an).long()

      image, an = image.to(device), an.to(device)

      #Forward passes
      predScrew = NetScrew(image)

      # calculate the loss
      lossFront = loss_fn(predScrew[:,0,:,:,:], an[:,0,:,:,:].squeeze())
      lossLat = loss_fn(predScrew[:,1,:,:,:], an[:,1,:,:,:].squeeze())
      loss = lossFront + lossLat
      loss.backward() #backward-pass
      
      optimizerScrew.step()
      #Set parameter gradients to zero.
      optimizerScrew.zero_grad()

      #Calculate average loss
      avg_train_lossScrew += loss.item() / len(train_loader)

    print('Epoch: ' + str(epoch))
    print('Train loss:')
    print(avg_train_lossScrew)
      #We can't call "evaluate split" below like with the test and validation set, because by then, we have already gone through 
      #he training data and going through it again would give new initializations of the training data, meaning the numbers would not be correct.
    #   with torch.no_grad():
    #     #Get the predicted segmentation
    #     pred_seg = torch.softmax(predScrew, dim=1).argmax(axis=1).unsqueeze(1)
        
    #     dices, IoUs, accuracies, sensitivities, specificities = segmentation_results(pred_seg.cpu(), an[:,1,:,:].cpu())

    #     avg_train_diceScrew += dices.mean().item() / len(train_loader)
    #     avg_train_IoUScrew  += IoUs.mean().item() / len(train_loader)
    #     avg_train_accScrew  += accuracies.mean().item() / len(train_loader)
    #     avg_train_sensScrew += sensitivities.mean().item() / len(train_loader)
    #     avg_train_specScrew += specificities.mean().item() / len(train_loader)
                                                                                                                     
    # print('Screw network:')
    # print('[%d] | Train      | Loss %.5f | Dice %.5f | IoU %.5f | Accuracy %.5f | Sensitivity %.5f | Specificity %.5f' % (epoch,
    #                                                                                                                       avg_train_lossScrew,
    #                                                                                                                       avg_train_diceScrew,
    #                                                                                                                       avg_train_IoUScrew,
    #                                                                                                                       avg_train_accScrew,
    #                                                                                                                       avg_train_sensScrew,
    #                                                                                                                       avg_train_specScrew))

    if return_results:
      epoch_results['epoch'] = epoch
      epoch_results['avg_train_loss'] = avg_train_loss
      epoch_results['avg_train_dice'] = avg_train_dice
      epoch_results['avg_train_IoU']  = avg_train_IoU
      epoch_results['avg_train_acc']  = avg_train_acc
      epoch_results['avg_train_sens'] = avg_train_sens
      epoch_results['avg_train_spec'] = avg_train_spec
    
    with open('wandb.token', 'r') as file:
            api_key = file.readline()
            wandb.login(key=api_key)

    wandb.init(project = "ort-project", entity = "ditteblom", reinit=True, name=run_name, settings=wandb.Settings(start_method="fork"))

    # Miscellaneous.
    if device.type == 'cuda':
        print("Training on GPU.")
    else:
        print("Training on CPU.")
        wandb.alert(
                        title=f"Training on {device}", 
                        text=f"Training on {device}"
                    )
    x_show = predScrew.squeeze()
    print('x_show shape')
    print(an.shape)
    print('output shape')
    print(x_show.shape)

    fig, axs = plt.subplots(2, 2, figsize = (20,20))
    axs[0,0].imshow(
        #image[0,:,0,:,:].permute(1, 2, 0).detach().cpu().numpy(), cmap = 'gray'
        an[0,0,0,:,:].squeeze().detach().cpu().numpy(), cmap = 'gray'
    )
    axs[0,0].set(title="Frontal image")

    axs[1,0].imshow(
        x_show[0,0,0,:,:].detach().cpu().numpy(),cmap = 'gray'
    )
    axs[1,0].set(title="Forward pass UNet")

    axs[0,1].imshow(
        #image[0,:,1,:,:].permute(1, 2, 0).detach().cpu().numpy(), cmap = 'gray'
        an[0,1,0,:,:].squeeze().detach().cpu().numpy(), cmap = 'gray'
    )
    axs[0,1].set(title="Lateral image")

    x_show = predScrew.squeeze()
    axs[1,1].imshow(
        x_show[0,0,1,:,:].detach().cpu().numpy(),cmap = 'gray'
    )
    axs[1,1].set(title="Forward pass UNet")

    wandb.log({"Train images": wandb.Image(fig)}, step=epoch)
    plt.close()

    # log on wandb.ai
    wandb.log({"epoch": epoch,
        "train loss": np.mean(avg_train_lossScrew),
    })
    # to watch on wandb.ai
    wandb.watch(NetScrew, log = "None")
                    
    # Save model checkpoint.
    state = {
      'epoch': epoch,
      'state_dict': NetScrew.state_dict(),
      'optimizer': optimizerScrew.state_dict()
    }
    save_name = run_name + '.ckpt'
    torch.save(state, save_name)

    if val_loader is not None:
      model.eval()
      with torch.no_grad():
        avg_val_loss, avg_val_dice, avg_val_IoU, avg_val_acc, avg_val_sens, avg_val_spec = evaluate_split(model, val_loader, loss_fn)
        print('[%d] | Validation | Loss %.5f | Dice %.5f | IoU %.5f | Accuracy %.5f | Sensitivity %.5f | Specificity %.5f' % (epoch,
                                                                                                                              avg_val_loss,
                                                                                                                              avg_val_dice,
                                                                                                                              avg_val_IoU,
                                                                                                                              avg_val_acc,
                                                                                                                              avg_val_sens,
                                                                                                                              avg_val_spec))
        if return_results:
          epoch_results['epoch'] = epoch
          epoch_results['avg_val_loss'] = avg_val_loss
          epoch_results['avg_val_dice'] = avg_val_dice
          epoch_results['avg_val_IoU']  = avg_val_IoU
          epoch_results['avg_val_acc']  = avg_val_acc
          epoch_results['avg_val_sens'] = avg_val_sens
          epoch_results['avg_val_spec'] = avg_val_spec
          
    if test_loader is not None:
      model.eval()
      with torch.no_grad():  
        avg_test_loss, avg_test_dice, avg_test_IoU, avg_test_acc, avg_test_sens, avg_test_spec = evaluate_split(model, test_loader, loss_fn)
        print('[%d] | Test       | Loss %.3f | Dice %.3f | IoU %.3f | Accuracy %.3f | Sensitivity %.3f | Specificity %.3f' % (epoch,
                                                                                                                              avg_test_loss,
                                                                                                                                avg_test_dice,
                                                                                                                              avg_test_IoU,
                                                                                                                              avg_test_acc,
                                                                                                                              avg_test_sens,
                                                                                                                              avg_test_spec))

        if return_results:
          epoch_results['avg_test_loss'] = avg_test_loss
          epoch_results['avg_test_dice'] = avg_test_dice
          epoch_results['avg_test_IoU']  = avg_test_IoU
          epoch_results['avg_test_acc']  = avg_test_acc
          epoch_results['avg_test_sens'] = avg_test_sens
          epoch_results['avg_test_spec'] = avg_test_spec

    if return_results:
      results = results.append(pd.DataFrame(epoch_results, index=[0]), ignore_index=True)

    if plot:
      assert val_loader is not None, 'We should get the validation dataset if plot=True.'
      #Show the first 6 items from the validation set.
      model.eval()  #Switch to evaluate mode.
      pred = torch.softmax(model(image), dim=1)

      #Draw 6 plots to look at. 
      plot_idxs = np.random.choice(np.arange(0, val_loader.__len__()), size=6, replace=False)

      plt.figure(figsize=(20,10))
      for k, val_idx in enumerate(plot_idxs):
        
        #Fetch validation data
        val_image, val_an0 = val_loader.dataset.__getitem__(val_idx)

        #Plot for the validation
        p = torch.softmax(model(val_image.unsqueeze(0).to(device)),dim=1).squeeze(0)[1].detach().cpu()
        real_mask = transforms.CenterCrop(rows-40)(val_an0).detach().cpu().numpy()

        plt.subplot(4,6,k+1)
        plt.imshow(val_image.squeeze(0))
        plt.axis('off')
        plt.title('Real image')

        plt.subplot(4,6,k+7)
        plt.imshow(p, cmap=plt.cm.gray) #First channel is the probability of a tumor
        plt.axis('off'); plt.clim([0,1])
        plt.title('Tumor probabilities')

        plt.subplot(4,6,k+13)
        plt.imshow(p > 0.5, cmap=plt.cm.gray) #Threshold @ 0.5
        plt.axis('off'); plt.clim([0,1])
        plt.title('Predicted segmentation')

        plt.subplot(4,6,k+19)
        plt.imshow(real_mask, cmap=plt.cm.gray)
        plt.axis('off'); plt.clim([0,1])
        plt.title('Real segmentation')
    
      plt.suptitle('Epoch [%d]\nTrain loss: %.3f, Train accuracy: %.3f\nValidation loss: %.3f, Validation accuracy: %.3f' % (epoch, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc), fontsize=15)
      plt.show()
    
  if return_results:
    return results

class CustomInvert:
  '''
  Rotates the data if need be.
  '''
  def __init__(self, invert):
    # initializing
    self.invert = invert

  def __call__(self, x):
    #assert len(x.shape) == 3, 'x should have [nchannel x rows x cols]'
    
    if self.invert == True:
      x = transforms.functional.invert(x)
    
    return x

class npStandardScaler:
  def fit(self, x):
    self.mean = np.mean(x)
    self.std = np.std(x)
  def transform(self, x):
    x -= self.mean
    x /= (self.std + 1e-7)
    return x

def saliency(input, model):
  '''
  Calculate the saliency.
  '''
  input = input.unsqueeze(0) #Add a batch dimension
  input = input.to(device)
  input.requires_grad = True

  model.eval()

  score = model(input)
  score, indices = torch.max(score, 1) # for classification problems

  #backward pass to get gradients of score predicted class w.r.t. input image
  score.backward()

  #get max along channel axis
  slc, _ = torch.max(torch.abs(input.grad[0]), dim=0)

  #normalize to [0..1]
  slc = (slc - slc.min())/(slc.max()-slc.min())

  #Detach
  im = input.detach().cpu().squeeze(0).numpy()
  slc = slc.cpu().numpy()

  return im, slc

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

