import torch
import math
import argparse 
import torch.nn.functional as F
from build_model.util import *
from build_model import *

def main():

    parser = argparse.ArgumentParser(description='Training vits and covits')

    # the class of the model, e.g., vit or covit  
    parser.add_argument('--net_name', type=str, default='vit', choices=['vit, covit'], 
                        help='the type of the neural network, e.g., vit or covit')

    # configuration for ViT 
    parser.add_argument('--in_channels', type=int, default=3, 
                        help='the number of input channles, e.g., 3')
    parser.add_argument('--img_size', nargs='+', type=int, default=[224, 224], 
                        help='the input image size, e.g., (224, 224)')
    parser.add_argument('--patch_size', nargs='+', type = int, default=[16, 16], 
                        help='the size of patches for patch embeddings, e.g., (16, 16)')
    parser.add_argument('--em_size', type=int, default=128, 
                        help='the embedding size, e.g., 512')

    parser.add_argument('--depth', type=int, default=4, 
                        help='the number basic blocks of vits and covits ')
    parser.add_argument('--d_K', type=int, default=128, 
                        help='the dimension of the Key')
    parser.add_argument('--d_V', type=int, default=128, 
                        help='the dimension of the Value')
    parser.add_argument('--num_heads', type=int, default=4, 
                        help='the number of heads')
    parser.add_argument('--att_drop_out', type=float, default=0., 
                        help='the drop_out rate for self attention')
    parser.add_argument('--MLP_expansion', type=int, default=4, 
                        help='the expansion rate for MLP layer in transformer encoder')
    parser.add_argument('--MLP_drop_out', type=float, default=0., 
                        help='the drop_out rate for MLP layers')

    parser.add_argument('--n_classes', type=int, default=10, 
                        help='the number of classes')

    # configuration for CoViT
    parser.add_argument('--kernel_size_group', nargs='+', type=int, default=[3,3,3,3], # equivelent to 4 heads 
                        help='the number of classes')
    parser.add_argument('--stride_group', nargs='+', type=int, default=[1,1,1,1], 
                        help='the number of classes')
    parser.add_argument('--padding_group', nargs='+', type=int, default=[1,1,1,1], 
                        help='the number of classes')

    # dataset configuration 
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        help='the data set to be trained with')
    parser.add_argument('--transform_resize', type=int, default=224, 
                        help='transform the inputs: resize the resolution')
    
    # training settings 
    parser.add_argument('--opt_name', type=str, default='sam', 
                        help='the name for optimizer')
    parser.add_argument('--lr', type=float, default=0.1, 
                        help='the learning rate for the opt')
    parser.add_argument('--momentum', type=float, default=0.9, 
                        help='the momentum for the opt')                        
    parser.add_argument('--train_start_epoch', type=int, default=1, 
                        help='the number of input channles')
    parser.add_argument('--train_end_epoch', type=int, default=4, 
                        help='the number of input channles')
    parser.add_argument('--train_ckp_freq', type=int, default=2, 
                        help='the frequence for saving network')
    parser.add_argument('--batch_size', type=int, default=256, 
                        help='the batch size for training and validating')

    # arguments for validation 
    parser.add_argument('--val_start_epoch', type=int, default=2, # notice that start_epoch for validation should be the same with val_ckp_freq
                        help='the number of input channles')
    parser.add_argument('--val_end_epoch', type=int, default=5, 
                        help='the number of input channles')
    parser.add_argument('--val_ckp_freq', type=int, default=2, # it must be an integer multiple of train_ckp_freq
                        help='the frequence for saving network')


    args = parser.parse_args()


    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the train loader, prepare for training 
    train_set, test_set = get_dataset(args)
    net_name, net = get_network(args) # net_name e.g., covit_D1_E512_K3_P16
    net = net.to(DEVICE)    

    optimizer = get_opt(args,net)
    lr_scheduler = get_lr_scheduler(args, optimizer, steps_per_epoch=math.ceil(len(train_set)/args.batch_size)) # step = len(train_loader)

    # initialization of the learner which is used to train and validate the network 
    learner = Learner((args.dataset, train_set, test_set), (net_name, net), optimizer, lr_scheduler, F.cross_entropy, device=DEVICE)
    learner.train(args.train_start_epoch, args.train_end_epoch, args.batch_size, args.train_ckp_freq)
    learner.validate(args.val_start_epoch, args.val_end_epoch, args.batch_size, args.val_ckp_freq)

if __name__ == '__main__':
    main()