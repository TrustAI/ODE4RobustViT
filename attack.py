import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from model_zoo.vit_homemade import ViT, ViT_merger
from model_zoo.covit import CoViT
from model_zoo.configs import *

import torch.nn.functional as F
from learner import Learner
import pandas as pd
import numpy

from torchattacks import PGD, FGSM, PGDL2, AutoAttack, CW
from utils import Save_dataset, Load_dataset, distance_calc
from tqdm import tqdm
import os 

BATCH_SIZE = 32
RESIZE = [224,224]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset configration test loader to be attacked 

test_CIFAR10 = datasets.CIFAR10(
    root='./dataset',
    train=False,
    download=True,
    transform=transforms.Compose([
    transforms.Resize(RESIZE),
    transforms.ToTensor(), 
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
    ])
)

test_loader = DataLoader(
    dataset=test_CIFAR10, batch_size=BATCH_SIZE
)

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

vit_homemade_conf_dict['vit_D4_E128_H1_R224_P32'] = vit_D4_E128_H1_R224_P32
vit_homemade_conf_dict['vit_D4_E128_H4_R224_P32'] = vit_D4_E128_H4_R224_P32
vit_homemade_conf_dict['vit_D8_E128_H1_R224_P32'] = vit_D8_E128_H1_R224_P32
vit_homemade_conf_dict['vit_D8_E128_H4_R224_P32'] = vit_D8_E128_H4_R224_P32

vit_merger_homemade_conf_dict['vit_merger_D4_E128_H1_R224_P32'] = vit_merger_D4_E128_H1_R224_P32
vit_merger_homemade_conf_dict['vit_merger_D4_E128_H4_R224_P32'] = vit_merger_D4_E128_H4_R224_P32
vit_merger_homemade_conf_dict['vit_merger_D8_E128_H1_R224_P32'] = vit_merger_D8_E128_H1_R224_P32
vit_merger_homemade_conf_dict['vit_merger_D8_E128_H4_R224_P32'] = vit_merger_D8_E128_H4_R224_P32


#covit_conf_dict['covit_D1_E512_K3_R224_P16'] = covit_D1_E512_K3_R224_P16
#covit_conf_dict['covit_D1_E512_K3333_R224_P16'] = covit_D1_E512_K3333_R224_P16
#covit_conf_dict['covit_D4_E512_K3_R224_P16'] = covit_D4_E512_K3_R224_P16
#covit_conf_dict['covit_D4_E512_K3333_R224_P16'] = covit_D4_E512_K3333_R224_P16
#covit_conf_dict['covit_D4_E512_K7777_R224_P16'] = covit_D4_E512_K7777_R224_P16
#covit_conf_dict['covit_D8_E512_K3_R224_P16'] = covit_D8_E512_K3_R224_P16
#covit_conf_dict['covit_D8_E512_K1357_R224_P16'] = covit_D8_E512_K1357_R224_P16
#covit_conf_dict['covit_D8_E512_K3333_R224_P16'] = covit_D8_E512_K3333_R224_P16
#ovit_conf_dict['covit_D8_E512_K7777_R224_P16'] = covit_D8_E512_K7777_R224_P16
#covit_conf_dict['covit_D8_E512_K3333_R224_P32'] = covit_D8_E512_K3333_R224_P32
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

# attack implement
def get_dataset(root):
    adv_dataset = Load_dataset(adv_folder=root,\
                        transform=transforms.Compose([
                        transforms.ToTensor()
                        ]))
    return adv_dataset

# parameters setting
switch = {
            'attacking': True,
            'summarization': True
         }

# setup attack parameters 
eps = 2 
alpha = 0.01 
steps = 20

#c = 1
#kappa = 0
#lr = 0.01
#steps = 500


# setup annotation 
#Lp_annotation = 'Linf'
Lp_annotation = 'L2'


#atk_annotation = 'fgsm'
atk_annotation = 'pgd'
#atk_annotation = 'cw'
#atk_annotation = 'aa'

#atk_par_annotation = f'eps{eps:.2f}'
atk_par_annotation = f'eps{eps:.2f}_alpha{alpha:.2f}_steps{steps}'
#atk_par_annotation = f'c{c}_kappa{kappa}_lr{lr}_steps{steps}'
#atk_par_annotation = f'eps{eps:.2f}'

# set T_max 
dist_T_list = [210, 220, 230, 240, 250, 260, 270, 280, 290, 300]

for net_name, net in vit_merger_homemade_dict.items():     

    # set up network
    epoch = 300
    net.load_state_dict(torch.load(f'./ckp_zoo/{net_name}_epoch{epoch}.pth'))
    net = net.eval().to(DEVICE)
    
    # root to save the attacked images      
    root=f'./dataset/adv_datasets/cifar10_testset_adv_{Lp_annotation}/{net_name}_epoch{epoch}/{atk_annotation}_{atk_par_annotation}/'

    if switch['attacking']:
        # set up attack  
        #atk = FGSM(net, eps)
        #atk = PGD(net, eps=eps, alpha=alpha, steps=steps)
        atk = PGDL2(net, eps=eps, alpha=alpha, steps=steps)
        #atk = CW(net,c=c, kappa=kappa, steps=steps,lr=lr)
        #atk = AutoAttack(net, norm='Linf', eps=eps)

        # cw attack has to include acc. distance for successful attack  
        if atk_annotation == 'cw':
            dist_sum_T = [0] * len(dist_T_list) 
            num_succ = [0] * len(dist_T_list)
            acc_sum_T = [0] * len(dist_T_list)

        saving_data = Save_dataset(root=root)
        index = 0 # index to name the perturbed image
        dist_sum = 0 
        for imgs, labels in tqdm(test_loader):
            print('attacking...\n')
            adv_imgs = atk(imgs, labels)
            
            # save the perturbed dataset 
            for img, label in zip(adv_imgs, labels):
                named_img = (str(index).zfill(8), img, label)
                saving_data.save_to_dataset(named_img)
                index += 1

            # calculate the distance while attacking 
            adv_imgs, imgs = adv_imgs.to(DEVICE), imgs.to(DEVICE)
            dist_vec = distance_calc(imgs.detach().clone() - adv_imgs.detach().clone(), Lp_norm=Lp_annotation)
            dist_sum += dist_vec.sum().cpu().numpy()

            # additional information 
            if atk_annotation == 'cw':
                learner = Learner(train_loader=None, \
                named_network=(net_name, net), optimizer=None, \
                loss_fn=F.cross_entropy, device=DEVICE
                )

                for i in range(len(dist_T_list)):
                    dist_mask = (dist_vec <= dist_T_list[i]).to(DEVICE)
                    num_succ[i] += dist_mask.sum().item() 
                    dist_sum_T[i] += dist_vec[dist_mask].to(DEVICE).sum().item()
                    _, acc_T = learner.evaluate(zip(adv_imgs[None,dist_mask], labels[None,dist_mask]))
                    acc_sum_T[i] += acc_T * len(adv_imgs[dist_mask]) 

        if atk_annotation == 'cw':            
            acc_avg_T = torch.tensor(acc_sum_T, device=DEVICE) / torch.tensor(num_succ, device=DEVICE)
            dist_avg_T = torch.tensor(dist_sum_T, device=DEVICE) / torch.tensor(num_succ, device=DEVICE)

        dist_avg = dist_sum/len(test_loader.dataset)
        saving_data.create_json_file()

    if switch['summarization']:
        # robust accuracy calculation 
        print('summarizing....')
        adv_CIFAR10 = get_dataset(root)
        adv_loader = DataLoader(dataset=adv_CIFAR10, batch_size=BATCH_SIZE)

        # calculate loss and accuracy for all adv_images
        learner = Learner(train_loader=None, \
                        named_network=(net_name, net), optimizer=None, \
                        loss_fn=F.cross_entropy, device=DEVICE
                        )
    
        loss, acc = learner.evaluate(adv_loader)

        # calculate the average distance and loss considering the maximum perturbation
        if not switch['attacking']:
            dist_sum = 0
            
            # cw attack has to include acc. distance for successful attack  
            if atk_annotation == 'cw':
                dist_sum_T = [0] * len(dist_T_list) 
                num_succ = [0] * len(dist_T_list)
                acc_sum_T = [0] * len(dist_T_list)
            
            for (imgs, labels), (adv_imgs, _labels) in tqdm(zip(test_loader, adv_loader)):
                
                assert (labels != _labels).sum() == 0, 'the order of the images in 2 datasets are not the same' 
                dist_vec = distance_calc(imgs - adv_imgs, Lp_norm=Lp_annotation)
                dist_sum += dist_vec.sum().detach().clone().cpu().numpy()
                
                # additional information should be included in cw attack 
                if atk_annotation == 'cw':
                    for i in range(len(dist_T_list)):
                        dist_mask = (dist_vec <= dist_T_list[i]).to(DEVICE)
                        num_succ[i] += dist_mask.sum().item() 
                        dist_sum_T[i] += dist_vec[dist_mask].to(DEVICE).sum().item()
                        _, acc_T = learner.evaluate(zip(adv_imgs[None,dist_mask], labels[None,dist_mask]))
                        acc_sum_T[i] += acc_T * len(adv_imgs[dist_mask]) 

            if atk_annotation == 'cw':            
                acc_avg_T = torch.tensor(acc_sum_T, device=DEVICE) / torch.tensor(num_succ, device=DEVICE)
                dist_avg_T = torch.tensor(dist_sum_T, device=DEVICE) / torch.tensor(num_succ, device=DEVICE)
            
            dist_avg = dist_sum/len(test_loader.dataset)
        
        # save the data 
        data_dict = {
                    'net_name':[net_name],
                    'epoch':[epoch],
                    'Lp':[Lp_annotation],
                    'attack':[atk_annotation],
                    'attack_parameters':[atk_par_annotation],
                    'loss':[loss],
                    'acc':[acc],
                    'avg_dist':[dist_avg]
                    }

        if atk_annotation == 'cw':
            for i in range(len(dist_T_list)):
                data_dict[f'acc.|T={dist_T_list[i]}'] = [acc_avg_T[i].item()]
                data_dict[f'avg_dist|T={dist_T_list[i]}'] = [dist_avg_T[i].item()]

        data_pd = pd.DataFrame(data_dict,index=None)
        
        if os.path.exists(f'./log_zoo/robust_testing_{atk_annotation}.csv'):
            header = None 
        else:
            header = True
        data_pd.to_csv(f'./log_zoo/robust_testing_{atk_annotation}.csv', mode='a', header=header)


