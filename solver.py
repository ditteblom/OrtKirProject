import torch
from models import MVCNN_VGG19, Baseline, VGG16, Inception_v3, MVCNN_Inception, MVCNN_UNet, MVCNN_VGG19_early, MVCNN_Baseline
import time
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import datetime
import os
import wandb
from torchvision import transforms
import matplotlib.pyplot as plt
from utils import saliency
from torch.nn.parallel import DistributedDataParallel

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class Solver(object):

    def __init__(self, train_loader, val_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Model configurations.
        self.repair = config.repair_type
        self.model = config.model
        self.fusion = config.fusion
        self.actfun = config.actfun
        self.alpha = config.alpha

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.lr = config.learning_rate
        
        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.log_step = config.log_step
        self.run_name = config.run_name

        # Build the model and tensorboard.
        self.build_model(self.model, self.actfun, self.alpha)

        #wandb setup
        with open('wandb.token', 'r') as file:
            api_key = file.readline()
            wandb.login(key=api_key)

        wandb.init(project = "ort-project", entity = "ditteblom", reinit=True, name=self.run_name, settings=wandb.Settings(start_method="fork"))

        # Miscellaneous.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            print("Training on GPU.")
        else:
            print("Training on CPU.")
            wandb.alert(
                            title=f"Training on {self.device}", 
                            text=f"Training on {self.device}"
                        )

        self.network.to(self.device)

        # initialize network on multiple GPUs
        setup(self.network, torch.cuda.device_count())

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # EX. dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.network = DistributedDataParallel(self.network, device_ids=list(range(torch.cuda.device_count())))

        # set up optmizer for network
        self.net_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.network.parameters()), self.lr)

        print('Training these layers:')
        for name,param in self.network.named_parameters():
            if param.requires_grad is True:
                print(name, param.requires_grad)

        # Set up weights and biases config
        wandb.config.update(config)

            
    def build_model(self, model, actfun, alpha):

        models_list = ['baseline','vgg16','mvcnn','mvcnn_baseline','mvcnn_vgg19','mvcnn_inception']
        self.model = model

        #assert (self.model in models_list), 'model needs to be baseline, vgg16, inceptionv3, mvcnn or mvcnnbaseline'

        if self.fusion == 'early':
            self.network = MVCNN_VGG19_early()
        else:
            if self.model == 'baseline':
                self.network = Baseline()
            if self.model == 'vgg16':
                self.network = VGG16()
            if self.model == 'inceptionv3':
                self.network = Inception_v3()
            if self.model == 'mvcnn_vgg19':
                self.network = MVCNN_VGG19()
            if self.model == 'mvcnn_inception':
                self.network = MVCNN_Inception()
            if self.model == 'mvcnn_baseline':
                self.network = MVCNN_Baseline(actfun, alpha)
            if self.model == 'mvcnn_unet':
                self.network = MVCNN_UNet()

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.net_optimizer.zero_grad()

#=====================================================================================================================================#
                
    def train(self):
        # Set data loader.
        data_loader = self.train_loader
        val_loader = self.val_loader
        
        # Print logs in specified order
        keys = ['train loss','validation loss']

        # to watch on wandb.ai
        wandb.watch(self.network, log = "None")
            
        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            train_loss = []

            # Fetch data.
            for _, (data, scores) in enumerate(data_loader):
                data, scores = data.to(self.device, dtype=torch.float), scores.to(self.device)#, grays.to(self.device), ans.to(self.device)

                #_, _, _, rows, _ = data.shape
                #Center crop the segmentation
                #ans = transforms.CenterCrop(rows-40)(ans).long()             
       
            # =================================================================================== #
            #                               2. Train the network                                  #
            # =================================================================================== #
            
                self.network = self.network.train()
                            
                # reser gradients
                self.reset_grad()

                # forward pass
                # if self.model == 'mvcnn_unet':
                #     output_recon, output = self.network(data)
                #     print('output_recon shape:')
                #     print(output_recon.shape)
                #     # loss
                #     loss_identic = F.l1_loss(output_recon.squeeze(), ans.squeeze())
                #     loss_score = F.mse_loss(output.float(), scores.float().unsqueeze(1))
                #     loss = loss_identic + loss_score
                # else:
                output = self.network(data)
                # loss
                loss = F.l1_loss(output.float(), scores.float().unsqueeze(1))

                # Backward and optimize.
                loss.backward()
                self.net_optimizer.step()

                # log the training loss
                train_loss.append(loss.detach().item())

            # =================================================================================== #
            #                               3. Validate the network                               #
            # =================================================================================== #
            
            val_loss = []

            # Fetch validation data.
            for data, scores in val_loader:

                data, scores = data.to(self.device, dtype=torch.float), scores.to(self.device)

                # centercrop annotation
                #ans = transforms.CenterCrop(88)(ans).long()

                # set netork to evaluation mode
                self.network = self.network.eval()

                with torch.no_grad():
                    # if self.model == 'mvcnn_unet':
                    #     output_recon, output = self.network(data)
                    #     # loss
                    #     loss_identic = F.l1_loss(output_recon.squeeze(), ans.squeeze())
                    #     loss_score = F.mse_loss(output.float(), target.float().unsqueeze(1))
                    #     loss = loss_identic + loss_score
                    # else:
                    output = self.network(data)
                    # loss
                    loss = F.l1_loss(output.float(), scores.float().unsqueeze(1))

                # log validation loss
                val_loss.append(loss.detach().item())

            # loss log
            loss_log = {}
            loss_log['train loss'] = np.mean(train_loss)
            loss_log['validation loss'] = np.mean(val_loss)


            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss_log[tag])
                print(log)

                # Save model checkpoint.
                state = {
                    'epoch': i+1,
                    'state_dict': self.network.state_dict(),
                    'optimizer': self.net_optimizer.state_dict(),
                    'activation function': self.actfun,
                }
                save_name = str(self.repair) + str(self.model) + '_'+self.run_name+'.pth'
                torch.save(state, save_name)

                #log images
                if self.model == 'mvcnn_unet':
                    fig, axs = plt.subplots(2, 2)
                    axs[0,0].imshow(
                        data[0,0,:,:,:].permute(1, 2, 0).detach().cpu().numpy(), cmap = 'gray'
                    )
                    axs[0,0].set(title="Frontal image")

                    x_show = output_recon.squeeze()
                    axs[1,0].imshow(
                        x_show[0,0,:,:].detach().cpu().numpy(),cmap = 'gray'
                    )
                    axs[1,0].set(title="Forward pass UNet")

                    axs[0,1].imshow(
                        data[0,1,:,:,:].permute(1, 2, 0).detach().cpu().numpy(), cmap = 'gray'
                    )
                    axs[0,1].set(title="Lateral image")

                    x_show = output_recon.squeeze()
                    axs[1,1].imshow(
                        x_show[0,1,:,:].detach().cpu().numpy(),cmap = 'gray'
                    )
                    axs[1,1].set(title="Forward pass UNet")

                    wandb.log({"Train images": wandb.Image(fig)}, step=i+1)
                    plt.close()
                else:
                    image = data[0]
                    score = scores[0]
                    _, model_slc = saliency(image, self.network)

                    fig, ax = plt.subplots(2,2, figsize = (10,10))
                    fig.suptitle(self.run_name)

                    ax[0,0].imshow(image.permute(1, 2, 3, 0)[0].cpu().numpy())
                    ax[0,0].set_title("Epoch: {:d}, Score: {:.4f}, Prediction: {:.4f}".format(i+1, score.item(), output[0].item()))
                    ax[0,0].imshow(model_slc[0], cmap = plt.cm.hot, alpha = 0.5)

                    ax[0,1].set_title('Saliency map frontal')
                    ax[0,1].imshow(model_slc[0], cmap = plt.cm.hot)

                    ax[1,0].imshow(image.permute(1, 2, 3, 0)[1].cpu().numpy())
                    ax[1,0].set_title("Epoch: {:d}, Score: {:.4f}, Prediction: {:.4f}".format(i+1, score.item(), output[0].item()))
                    ax[1,0].imshow(model_slc[1], cmap = plt.cm.hot, alpha = 0.5)

                    ax[1,1].set_title('Saliency map lateral')
                    ax[1,1].imshow(model_slc[1], cmap = plt.cm.hot)

                    wandb.log({"Train images": wandb.Image(fig)}, step=i+1)
                    plt.close()

            
            # log on wandb.ai
            wandb.log({"epoch": i+1,
                        "train loss": np.mean(train_loss),
                        "val loss": np.mean(val_loss),
            })