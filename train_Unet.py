import numpy as np
from dataloader import get_loader
from utils import train_anno
from models import UNet
from torchvision.transforms import transforms
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

repair_type = "001_hansson_pin_system"
size = 200
batch_size = 64 # 1 in UNet paper

train_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((size,size)),
                                    transforms.RandomChoice([transforms.GaussianBlur(kernel_size=3, sigma=(1, 1)),
                                    transforms.RandomRotation(degrees=(15))]),
                                    ])
test_transform = val_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((size,size)),
                                    ])

train_loader = get_loader(repair_type, split = 'train', data_path = "Data", batch_size=batch_size, transform = train_transform, num_workers=0)
#val_loader = get_loader(repair_type, split = 'val', data_path = "Data", batch_size=batch_size, transform = val_transform, num_workers=0)
#test_loader = get_loader(repair_type, split = 'test', data_path = "Data", batch_size=batch_size, transform = test_transform, num_workers=0)

model = UNet()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
decayRate = 0.96
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

num_epochs = 10000
loss_fun = nn.CrossEntropyLoss()

print('Starting training...')
out_dict = train_anno(train_loader, epochs=num_epochs, plot=False, return_results=False)