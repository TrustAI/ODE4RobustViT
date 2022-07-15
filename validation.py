import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from model_zoo.vit_homemade import ViT
from model_zoo.covit import CoViT
from model_zoo.configs import *

import torch.nn.functional as F
from learner import Learner
import pandas as pd
from utils import get_loader
import argparse 

#BATCH_SIZE = 1024
#RESIZE = [224,224]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset configration 
#train_CIFAR10 = datasets.CIFAR10(
#    root='./dataset',
#    train=True,
#    download=True,
#    transform=transforms.Compose([
#    transforms.Resize(RESIZE),
#    transforms.ToTensor(), 
#    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
#    ])
#)

#test_CIFAR10 = datasets.CIFAR10(
#    root='./dataset',
#    train=False,
#    download=True,
#    transform=transforms.Compose([
#    transforms.Resize(RESIZE),
#    transforms.ToTensor(), 
#    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
#    ])
#)

#train_loader = DataLoader(
#    dataset=train_CIFAR10, batch_size=BATCH_SIZE
#    )
#test_loader = DataLoader(
#    dataset=test_CIFAR10, batch_size=BATCH_SIZE
#)

# model setup
vit_homemade_conf_dict = {}
covit_conf_dict = {}

vit_homemade_dict = {}
covit_dict = {}

#vit_homemade_conf_dict['vit_D1_E512_H1_R224_P16'] = vit_D1_E512_H1_R224_P16
#vit_homemade_conf_dict['vit_D1_E512_H4_R224_P16'] = vit_D1_E512_H4_R224_P16
#vit_homemade_conf_dict['vit_D4_E512_H1_R224_P16'] = vit_D4_E512_H1_R224_P16
#vit_homemade_conf_dict['vit_D4_E512_H4_R224_P16'] = vit_D4_E512_H4_R224_P16
#vit_homemade_conf_dict['vit_D8_E512_H1_R224_P16'] = vit_D8_E512_H1_R224_P16
#vit_homemade_conf_dict['vit_D8_E512_H4_R224_P16'] = vit_D8_E512_H4_R224_P16
#vit_homemade_conf_dict['vit_D8_E512_H4_R224_P32'] = vit_D8_E512_H4_R224_P32
#vit_homemade_conf_dict['vit_D12_E512_H8_R224_P32'] = vit_D12_E512_H8_R224_P32
vit_homemade_conf_dict['vit_D4_E128_H1_R224_P32'] = vit_D4_E128_H1_R224_P32
vit_homemade_conf_dict['vit_D4_E128_H4_R224_P32'] = vit_D4_E128_H4_R224_P32
vit_homemade_conf_dict['vit_D8_E128_H1_R224_P32'] = vit_D8_E128_H1_R224_P32
vit_homemade_conf_dict['vit_D8_E128_H4_R224_P32'] = vit_D8_E128_H4_R224_P32



#covit_conf_dict['covit_D1_E512_K3_R224_P16'] = covit_D1_E512_K3_R224_P16
#covit_conf_dict['covit_D1_E512_K3333_R224_P16'] = covit_D1_E512_K3333_R224_P16
#covit_conf_dict['covit_D4_E512_K3_R224_P16'] = covit_D4_E512_K3_R224_P16
#covit_conf_dict['covit_D4_E512_K3333_R224_P16'] = covit_D4_E512_K3333_R224_P16
#covit_conf_dict['covit_D4_E512_K7777_R224_P16'] = covit_D4_E512_K7777_R224_P16
#covit_conf_dict['covit_D8_E512_K3_R224_P16'] = covit_D8_E512_K3_R224_P16
#covit_conf_dict['covit_D8_E512_K1357_R224_P16'] = covit_D8_E512_K1357_R224_P16
#covit_conf_dict['covit_D8_E512_K3333_R224_P16'] = covit_D8_E512_K3333_R224_P16
#covit_conf_dict['covit_D8_E512_K7777_R224_P16'] = covit_D8_E512_K7777_R224_P16
#ovit_conf_dict['covit_D8_E512_K3333_R224_P32'] = covit_D8_E512_K3333_R224_P32
#covit_conf_dict['covit_D12_E512_4xK3_4xK5_R224_P32'] = covit_D12_E512_4xK3_4xK5_R224_P32
#covit_conf_dict['covit_D16_E512_4xK3_4xK5_R224_P32'] = covit_D16_E512_4xK3_4xK5_R224_P32
covit_conf_dict['covit_D4_E128_K3_R224_P32'] = covit_D4_E128_K3_R224_P32
covit_conf_dict['covit_D4_E128_K3333_R224_P32'] = covit_D4_E128_K3333_R224_P32
covit_conf_dict['covit_D8_E128_K3_R224_P32'] = covit_D8_E128_K3_R224_P32
covit_conf_dict['covit_D8_E128_K3333_R224_P32'] = covit_D8_E128_K3333_R224_P32


for net_name, conf in vit_homemade_conf_dict.items():
    vit_homemade_dict[net_name] = ViT(
                                    in_channels=conf['embedding']['in_channels'],
                                    patch_size=conf['embedding']['patch_size'],
                                    em_size=conf['embedding']['em_size'],
                                    img_size=conf['embedding']['img_size'],
                                    depth=conf['encoder']['depth'],
                                    n_classes=conf['cls_head']['n_classes'],
                                    forward_expansion = conf['encoder']['MLP_expansion'],
                                    forward_drop_out = conf['encoder']['MLP_drop_out'],
                                    d_K = conf['encoder']['d_K'],
                                    d_V = conf['encoder']['d_V'],
                                    num_heads = conf['encoder']['num_heads'],
                                    drop_out = conf['encoder']['att_drop_out']
                                    ) 

for net_name, conf in covit_conf_dict.items():
    covit_dict[net_name] = CoViT(
                                    in_channels=conf['embedding']['in_channels'],
                                    patch_size=conf['embedding']['patch_size'],
                                    em_size=conf['embedding']['em_size'],
                                    img_size=conf['embedding']['img_size'],
                                    depth=conf['encoder']['depth'],
                                    n_classes=conf['cls_head']['n_classes'],
                                    forward_expansion = conf['encoder']['MLP_expansion'],
                                    forward_drop_out = conf['encoder']['MLP_drop_out'],
                                    kernel_size_group = conf['encoder']['kernel_size_group'],
                                    stride_group = conf['encoder']['stride_group'],
                                    padding_group = conf['encoder']['padding_group'],
                                    ) 

def validation(net_dict,train_loader, test_loader, epoch_interval, save_frequency = 5):

    # evaluation 
    epoch_start = epoch_interval[0]
    epoch_end = epoch_interval[1]
    epoch_list = [i for i in range(epoch_start, epoch_end) if i%save_frequency == 0]

    n_start = epoch_start // save_frequency
    #epoch_list = [445, 450]
    for net_name, net in net_dict.items():
        n = n_start
        for epoch in epoch_list:
            net.load_state_dict(torch.load(f'./ckp_zoo/{net_name}_epoch{epoch}.pth'))
            net = net.to(DEVICE)
            learner = Learner(train_loader, \
                            named_network=(net_name, net), optimizer=None, \
                            loss_fn=F.cross_entropy, device=DEVICE
                            )
            train_loss, train_acc = learner.evaluate(train_loader)
            val_loss, val_acc = learner.evaluate(test_loader)
            data_pd = pd.DataFrame(
                                {
                                'epoch':[epoch],
                                'train_loss':[train_loss],
                                'train_acc':[train_acc],
                                'val_loss':[val_loss],
                                'val_acc':[val_acc]
                                }, index=[n])
            if n >0: 
                header = None
            else: 
                header = True

            data_pd.to_csv(f'./log_zoo/training_results_{net_name}.csv', mode='a', header=header)
            n += 1


def main():
    parser = argparse.ArgumentParser(description='training the neural network')

    parser.add_argument('--dataset', type=str, default='cifar10', 
                        help='the data set to be trained with')
    parser.add_argument('--transform_resize', type=int, default=224, 
                        help='transform the inputs: resize the resolution')                    
    parser.add_argument('--train_batch_size', type=int, default=256, 
                        help='training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=256, 
                        help='evaluating batch size')

    args = parser.parse_args()
    train_loader, test_loader = get_loader(args)
    
    validation(covit_dict,train_loader, test_loader, epoch_interval=(300, 301))
    validation(vit_homemade_dict,train_loader, test_loader, epoch_interval=(300, 301))

if __name__ == '__main__':
    main()