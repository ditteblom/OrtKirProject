import torch
from models import Baseline, VGG16, Inception_v3
import time
import torch.nn.functional as F
import numpy as np
import datetime
import wandb

wandb.init(project = "ort-project", entity = "ditteblom", reinit=True)

class Solver(object):

    def __init__(self, train_loader, val_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Model configurations.
        self.repair = config.repair_type
        self.model = config.model

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.lr = config.learning_rate
        
        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.log_step = config.log_step

        # Build the model and tensorboard.
        self.build_model(self.model)

            
    def build_model(self, model):

        models_list = ['baseline','vgg16','inceptionv3']
        self.model = model

        assert (self.model in models_list), 'model needs to be baseline, vgg16 or inceptionv3'

        if self.model == 'baseline':
            self.network = Baseline()
        if self.model == 'vgg16':
            self.network = VGG16()
        if self.model == 'inceptionv3':
            self.network = Inception_v3()
    
        self.net_optimizer = torch.optim.Adam(self.network.parameters(), self.lr)
        
        self.network.to(self.device)

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
            
        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            train_loss = []

            # Fetch data.
            for minibatch_no, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                        
       
            # =================================================================================== #
            #                               2. Train the network                                  #
            # =================================================================================== #
            
                self.network = self.network.train()
                            
                # reser gradients
                self.reset_grad()

                # forward pass
                output = self.network(data)  
                
                # loss
                loss = F.mse_loss(output.float(), target.float())

                # Backward and optimize.
                loss.backward()
                self.net_optimizer.step()

                # log the training loss
                train_loss.append(loss.item())

            # Add training loss to log.
            loss_log = {}
            loss_log['train loss'] = np.mean(train_loss)

            # =================================================================================== #
            #                               3. Validate the network                               #
            # =================================================================================== #
            
            val_loss = []

            # Fetch validation data.
            for data, target in val_loader:

                data, target = data.to(self.device), target.to(self.device)

                # set netork to evaluation mode
                self.network = self.network.eval()

                with torch.no_grad():
                    output = self.network(data)

                # log validation loss
                val_loss.append(F.mse_loss(output, target).cpu().item())

            # add validation loss to log.
            loss_log['validation loss'] = np.mean(val_loss)


            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # log on wandb.ai
            wandb.log({"train loss": np.mean(train_loss),
                        "val loss": np.mean(val_loss),
            })

            # to watch on wandb.ai
            wandb.watch(self.network, log = "None")

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss_log[tag])
                print(loss_log)

                # Save model checkpoint.
                state = {
                    'epoch': i+1,
                    'state_dict': self.network.state_dict(),
                }
                save_name = 'model_checkpoint_' + str(self.model) + '.pth'
                torch.save(state, save_name)