import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.dataloader import DataLoader
from learner import Learner

from model_zoo.vit_homemade import ViT
from model_zoo.covit import CoViT
from model_zoo.configs import *
from utils import get_loader, get_network
import argparse 


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    parser = argparse.ArgumentParser(description='training the neural network')
    parser.add_argument('--net_name', type=str, 
                        help='the model to be trained')
    parser.add_argument('--net_type', choices=['vit', 'covit'], 
                        help='the model to be trained')

    parser.add_argument('--dataset', type=str, default='cifar10', 
                        help='the data set to be trained with')
    parser.add_argument('--input_size', type=int, default=32, 
                        help='the size of input to be specified')
    parser.add_argument('--start_epoch', type=int, default=0, 
                        help='triaining epochs')
    parser.add_argument('--end_epoch', type=int, default=151, 
                        help='triaining epochs')
    parser.add_argument('--batch_size', type=int, default=256, 
                        help='batch size')
    parser.add_argument('--continue_train', default=False, action='store_true', 
                        help='momentum for training')


    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    

    args = parser.parse_args()
    


def train(args, model):

    train_loader, test_loader = get_loader(args)

    net = get_network(args.net_name)

    if args.continue_train:
        net.load_state_dict(torch.load(f'./ckp_zoo/{args.net_name}_epoch{args.epoch_start}.pth'))

    net = net.to(DEVICE)
    learner = Learner(train_loader=train_loader, named_network=(args.net_name, net), \
                    optimizer=optimizer, lr_scheduler=lr_scheduler, loss_fn=F.cross_entropy, device=DEVICE)

    learner.train(epochs=(args.epoch_start, args.epoch_end)) 



if __name__ == '__main__':
    main()