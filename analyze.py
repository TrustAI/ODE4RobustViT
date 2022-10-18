import torch 
import argparse 

from build_model.util import *
from analyze_model.util import *
from analyze_model import *


def main():
    parser = argparse.ArgumentParser(description='analysis the trained neural network')

    # choose the model 
    parser.add_argument('--net_name', type=str, default='vit', 
                        help='the name of neural network, e.g., vit or covit')
    parser.add_argument('--net_epoch', type=int, default=4, 
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
    parser.add_argument('--num_imgs', type=int, default=1, 
                        help='the number of images to be analysed')

    # arguments for sigular value calculation 
    parser.add_argument('--approximation', action='store_true', default=False,
                        help='estimate the maximum singular value instead of actually compute it',)

    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, test_set = get_dataset(args)

    # set up network
    net_name, net = get_network(args) # net_name e.g., covit_D1_E512_K3_P1
    net_name = net_name + f'_epoch({args.net_epoch})'
    net.load_state_dict(torch.load(f'./build_model/ckp/{args.dataset}/{net_name}.pth'))
    net = net.to(DEVICE)

    analysis = Analysis((args.dataset, test_set), (net_name, net), DEVICE)
    analysis.analyze(args.num_imgs, [[1], [0, 1, 0]], args.approximation)

if __name__ == '__main__':
    main()



