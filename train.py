import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.dataloader import DataLoader
from learner import Learner

from model_zoo.vit_homemade import ViT, ViT_merger
from model_zoo.covit import CoViT, CoViT_merger
from model_zoo.configs import *
from optimizers.opt_sam import SAM

from utils import get_loader
import argparse 

    



# hyper-parameters 
EPOCHS = 301
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#BATCH_SIZE = 256
#RESIZE = [224,224]

lr = 0.1
momentum = 0.9
# dataset 
#train_CIFAR10 = datasets.CIFAR10(
#    root='./dataset',
#    train=True,
#    download=True,
#    transform=transforms.Compose([
#    transforms.RandomHorizontalFlip(),
#    transforms.RandomCrop(32, padding=4),
#    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
#    dataset=train_CIFAR10, batch_size=BATCH_SIZE, shuffle=True
#    )
#test_loader = DataLoader(
#    dataset=test_CIFAR10, batch_size=BATCH_SIZE
#)



# model setup
vit_homemade_conf_dict = {}
vit_merger_homemade_conf_dict = {}

covit_conf_dict = {}
covit_merger_conf_dict = {}

vit_homemade_dict = {}
vit_merger_homemade_dict = {}

covit_dict = {}
covit_merger_dict = {}

#vit_homemade_conf_dict['vit_D1_E512_H1_R224_P16'] = vit_D1_E512_H1_R224_P16
#vit_homemade_conf_dict['vit_D1_E512_H4_R224_P16'] = vit_D1_E512_H4_R224_P16
#vit_homemade_conf_dict['vit_D4_E512_H1_R224_P16'] = vit_D4_E512_H1_R224_P16
#vit_homemade_conf_dict['vit_D4_E512_H4_R224_P16'] = vit_D4_E512_H4_R224_P16
#vit_homemade_conf_dict['vit_D8_E512_H1_R224_P16'] = vit_D8_E512_H1_R224_P16
#vit_homemade_conf_dict['vit_D8_E512_H4_R224_P16'] = vit_D8_E512_H4_R224_P16
#vit_homemade_conf_dict['vit_D8_E512_H4_R224_P32'] = vit_D8_E512_H4_R224_P32
#vit_homemade_conf_dict['vit_D12_E512_H8_R224_P32'] = vit_D12_E512_H8_R224_P32
#vit_homemade_conf_dict['vit_D4_E128_H1_R224_P32'] = vit_D4_E128_H1_R224_P32
#vit_homemade_conf_dict['vit_D4_E128_H4_R224_P32'] = vit_D4_E128_H4_R224_P32
#vit_homemade_conf_dict['vit_D8_E128_H1_R224_P32'] = vit_D8_E128_H1_R224_P32
#vit_homemade_conf_dict['vit_D8_E128_H4_R224_P32'] = vit_D8_E128_H4_R224_P32

#vit_merger_homemade_conf_dict['vit_merger_D4_E128_H1_R224_P32'] = vit_merger_D4_E128_H1_R224_P32
#vit_merger_homemade_conf_dict['vit_merger_D4_E128_H4_R224_P32'] = vit_merger_D4_E128_H4_R224_P32
#vit_merger_homemade_conf_dict['vit_merger_D8_E128_H1_R224_P32'] = vit_merger_D8_E128_H1_R224_P32
#vit_merger_homemade_conf_dict['vit_merger_D8_E128_H4_R224_P32'] = vit_merger_D8_E128_H4_R224_P32


#covit_conf_dict['covit_D1_E512_K3_R224_P16'] = covit_D1_E512_K3_R224_P16
#covit_conf_dict['covit_D1_E512_K3333_R224_P16'] = covit_D1_E512_K3333_R224_P16
#covit_conf_dict['covit_D4_E512_K3_R224_P16'] = covit_D4_E512_K3_R224_P16
#covit_conf_dict['covit_D4_E512_K3333_R224_P16'] = covit_D4_E512_K3333_R224_P16
#covit_conf_dict['covit_D4_E512_K7777_R224_P16'] = covit_D4_E512_K7777_R224_P16
#covit_conf_dict['covit_D8_E512_K3_R224_P16'] = covit_D8_E512_K3_R224_P16
#covit_conf_dict['covit_D8_E512_K1357_R224_P16'] = covit_D8_E512_K1357_R224_P16
#covit_conf_dict['covit_D8_E512_K3333_R224_P16'] = covit_D8_E512_K3333_R224_P16
#covit_conf_dict['covit_D8_E512_K7777_R224_P16'] = covit_D8_E512_K7777_R224_P16
#covit_conf_dict['covit_D8_E512_K3333_R224_P32'] = covit_D8_E512_K3333_R224_P32
#covit_conf_dict['covit_D12_E512_4xK3_4xK5_R224_P32'] = covit_D12_E512_4xK3_4xK5_R224_P32
#covit_conf_dict['covit_D16_E512_4xK3_4xK5_R224_P32'] = covit_D16_E512_4xK3_4xK5_R224_P32
#covit_conf_dict['covit_D4_E128_K3_R224_P32'] = covit_D4_E128_K3_R224_P32
covit_conf_dict['covit_D4_E128_K3333_R224_P32'] = covit_D4_E128_K3333_R224_P32
covit_conf_dict['covit_D8_E128_K3_R224_P32'] = covit_D8_E128_K3_R224_P32
covit_conf_dict['covit_D8_E128_K3333_R224_P32'] = covit_D8_E128_K3333_R224_P32

#covit_merger_conf_dict['covit_merger_D4_E128_K3_R224_P32'] = covit_merger_D4_E128_K3_R224_P32
#covit_merger_conf_dict['covit_merger_D4_E128_K3333_R224_P32'] = covit_merger_D4_E128_K3333_R224_P32
#covit_merger_conf_dict['covit_merger_D8_E128_K3_R224_P32'] = covit_merger_D8_E128_K3_R224_P32
#covit_merger_conf_dict['covit_merger_D8_E128_K3333_R224_P32'] = covit_merger_D8_E128_K3333_R224_P32



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

for net_name, conf in vit_merger_homemade_conf_dict.items():
    vit_merger_homemade_dict[net_name] = ViT_merger(
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

for net_name, conf in covit_merger_conf_dict.items():
    covit_merger_dict[net_name] = CoViT_merger(
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


if __name__ == '__main__':


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

    net_dict = {}
    net_dict.update(vit_homemade_dict)
    net_dict.update(vit_merger_homemade_dict)
    net_dict.update(covit_dict)
    net_dict.update(covit_merger_dict)

    continue_train = False
    epoch_start = 0
    epoch_end = 301

    base_optimizer =torch.optim.SGD
    for net_name, net in net_dict.items():

        if continue_train:
            net.load_state_dict(torch.load(f'./ckp_zoo/{net_name}_epoch{epoch_start}.pth'))

        net = net.to(DEVICE)
        optimizer = SAM(net.parameters(), base_optimizer, lr=lr, momentum=momentum)
        #optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        learner = Learner(train_loader=train_loader, named_network=(net_name, net), \
                        optimizer=optimizer, loss_fn=F.cross_entropy, device=DEVICE)
        learner.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                                                steps_per_epoch=len(train_loader),epochs=EPOCHS
                                                                )
        if continue_train:
            learner.train(epochs=(epoch_start, epoch_end)) 
        else:
            learner.train(epochs=(EPOCHS,))


