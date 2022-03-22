import argparse
from solver import Solver
from dataloader import get_loader
import torchvision.transforms as transforms


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.

    # transforms
    if config.model == 'baseline':
        size = 129
    if config.model == 'vgg16':
        size = 225
    if config.model == 'inceptionv3':
        size = 300

    train_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((size,size)),
                                    #transforms.Grayscale(num_output_channels = 1),
                                    transforms.RandomChoice([transforms.GaussianBlur(kernel_size=3, sigma=(1, 1)),
                                    transforms.RandomRotation(degrees=(15)),
                                    ])])
    val_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((size,size)),
                                    #transforms.Grayscale(num_output_channels = 1),
                                    ])

    # Data loaders.
    train_loader = get_loader(config.repair_type, 'train', config.data_dir, config.batch_size, transform = train_transform)
    val_loader = get_loader(config.repair_type, 'val', config.data_dir, config.batch_size, transform = val_transform)

    solver = Solver(train_loader, val_loader, config)

    solver.train()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--repair_type', type=str, default='001_hansson_pin_system', help='type of fracture repair')
    parser.add_argument('--model', type=str, default='inceptionv3', help='type of model')
    
    # Training configuration.
    parser.add_argument('--data_dir', type=str, default='/work3/dgro/Data/')
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=1000, help='number of total iterations')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate for optimizer')
    
    # Miscellaneous.
    parser.add_argument('--log_step', type=int, default=1)

    config = parser.parse_args()
    print(config)
    main(config)