import torch
import torch.nn.functional as F
import argparse 

from attack_model import * 
from attack_model.util import * 
from build_model.util import * 


def main():
    parser = argparse.ArgumentParser(description='attacking the trained neural network')

    # choose the model 
    parser.add_argument('--net_name', type=str, default='vit', 
                        help='the name of neural network, e.g., vit or covit')
    parser.add_argument('--net_epoch', type=int, default=8, 
                        help='the epoch of the model')

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
    parser.add_argument('--kernel_size_group', nargs='+', type=int, default=[3,3,3,3], 
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
    parser.add_argument('--batch_size', type=int, default=256, 
                        help='the batch size for training and testing')
    parser.add_argument('--fraction_size', type=float, default=512, 
                        help='the subset of the test data to be attacked')

    # attacking settings 
    parser.add_argument('--att_method', type=str, default='pgd', 
                        help='attactink method')
    parser.add_argument('--att_norm', type=str, default='L2', 
                        help='the norm for attacking')                        
    parser.add_argument('--num_saving', type=int, default=10, 
                        help='number of images to be saved')                        

    # attacking setting for fgsm and aa
    parser.add_argument('--epsilon', type=float, default=2., 
                        help='epsilon for fgsm')

    # additional attacking setting for pgd 
    parser.add_argument('--alpha', type=float, default=0.01, 
                        help='alpha for pgd')
    parser.add_argument('--steps', type=int, default=20, 
                        help='steps for pgd')                        

    # additional attacking setting for cw 
    parser.add_argument('--c', type=float, default=1, 
                        help='c for cw attack')
    parser.add_argument('--kappa', type=float, default=0., 
                        help='kappa for cw attack')                        
    parser.add_argument('--lr', type=float, default=0.01, 
                        help='kappa for cw attack')                        

    args = parser.parse_args()
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up testloader 
    _, test_set = get_dataset(args)

    # set up network
    net_name, net = get_network(args) # net_name e.g., covit_D1_E512_K3_P1
    net_name = net_name + f'_epoch({args.net_epoch})'
    net.load_state_dict(torch.load(f'./build_model/ckp/{args.dataset}/{net_name}.pth'))
    net = net.to(DEVICE)

    # set up attacking 
    attack_name, attack = get_attack(net,args)

    attack_ensembles = Attack_Ensembles(named_dataset=(args.dataset, test_set), named_network=(net_name, net), named_attack=(attack_name, attack), \
                     Lp_for_dist=['Linf', 'L2'], num_saving=args.num_saving, loss_fn=F.cross_entropy, device=DEVICE)
    attack_ensembles.attack(args.batch_size, args.fraction_size)

if __name__ == '__main__':
    main()