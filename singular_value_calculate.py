from eagerpy import reshape
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from utils import model_decompose
import torch 
import pandas as pd
import os
from utils import get_loader
import argparse 
from tqdm import tqdm

from model_zoo.configs import *
from model_zoo.vit_homemade import ViT
from model_zoo.covit import CoViT

import model_zoo.vit_homemade
import model_zoo.covit

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

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
#vit_homemade_conf_dict['vit_D4_E128_H1_R224_P32'] = vit_D4_E128_H1_R224_P32

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


def max_svd_compute(jacobian_matrix, top_k = 1):
    '''
    input:  jacobian maxtrix of size 2
    return: k largest singular value 
    '''
    svdvals = torch.linalg.svdvals(jacobian_matrix)
    return torch.topk(svdvals, top_k)[0]


import torch 
import torch.nn as nn

def simga_max_estimate(func, x):
    '''
    x is a matrix of size (B,N,D)
    output is the frobenius norm, infty norm and L_1 norm of the jacobian w.r.t. inputs 
    '''
    _x = x.detach().clone().requires_grad_()
    
    y = func(_x)

    frobenius_norm = 0
    row_abs_sum = []
    col_abs_sum = 0
    for i in tqdm(range(_x.shape[1])):
        for j in range(_x.shape[2]):
            grad = torch.autograd.grad(inputs=_x, outputs=y[0,i,j], retain_graph=True)[0].detach().clone()

            frobenius_norm += grad.square().sum() 
            row_abs_sum.append(grad.abs().sum()) # infinity norm 
            col_abs_sum += grad.abs() # L_1 norm 

    row_abs_sum = torch.tensor(row_abs_sum)
    return frobenius_norm.sqrt(), row_abs_sum.max(), col_abs_sum.max()


def main():

    parser = argparse.ArgumentParser(description='training the neural network')

    parser.add_argument('--dataset', type=str, default='cifar10', 
                        help='the data set to be trained with')
    parser.add_argument('--transform_resize', type=int, default=224, 
                        help='transform the inputs: resize the resolution')
    parser.add_argument('--train_batch_size', type=int, default=1, 
                        help='training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=128, 
                        help='evaluating batch size')
    parser.add_argument('--trained_epoch', type=int, default=300, 
                        help='the evaluated epoch of network')

    parser.add_argument('--num_imgs', type=int, default=50, 
                        help='the number of images to be analysed')
    parser.add_argument('--estimation', action='store_true', default=False,
                        help='estimate the maximum singular value instead of actually compute it',)

    args = parser.parse_args()
    train_loader, test_loader = get_loader(args)
    
    net_dict = {}
    net_dict.update(vit_homemade_dict)
    net_dict.update(covit_dict)

    for net_name, net in net_dict.items():

        # load the model         
        net.load_state_dict(torch.load(f'./ckp_zoo/{net_name}_epoch{args.trained_epoch}.pth'))
        net = net.to(DEVICE)
        

        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            # correctly classified sub imgs and labels sets 
            index_corrects = net(imgs).argmax(dim=1) == labels
            imgs_sub, labels_sub = imgs[index_corrects], labels[index_corrects]

            for i in tqdm(range(args.num_imgs)):
                features_list = []
                
                if net_name[:5] == 'covit':
                    embedding, encoders, head = model_decompose(net, model_zoo.covit)
                else:    
                    embedding, encoders, head = model_decompose(net, model_zoo.vit_homemade) 
                with torch.no_grad():
                    features_list.append(embedding(imgs_sub[i:i+1])) # feature map for embedding 
                    for j, encoder in enumerate(encoders): # feature map for each encoder
                        features_list.append(encoder(features_list[j])) 
                    features_list.append(head(features_list[-1])) # feature map for classification head 
                # check the result 
                pred = features_list[-1].argmax(dim=1)
                assert pred == labels_sub[i], "something wrong"

                # maximum singular value computation for encoders 
                if args.estimation:
                    frobenius_norm_list = []
                    L_infty_norm_list = []
                    L_1_norm_list = []
                    for k, encoder in enumerate(encoders): # feature map for each encoder
                        frobenius_norm, L_infty_norm, L_1_norm = simga_max_estimate(encoder, features_list[k]) 
                        frobenius_norm_list.append(frobenius_norm)
                        L_infty_norm_list.append(L_infty_norm)
                        L_1_norm_list.append(L_1_norm)

                    est_detail_dict = {}
                    for n in range(len(frobenius_norm_list)):
                        est_detail_dict[f'F-layer{n+1}'] = [frobenius_norm_list[n].item()]
                        est_detail_dict[f'L_infty-layer{n+1}'] = [L_infty_norm_list[n].item()]
                        est_detail_dict[f'L_1-layer{n+1}'] = [L_1_norm_list[n].item()]
                    
                    est_detail_pd = pd.DataFrame(est_detail_dict)

                    if  os.path.exists(f'./log_zoo/singular_val_estimation/cifar10/{net_name}_epoch{args.trained_epoch}.csv') or i > 0: 
                        header = None
                    else: 
                        header = True

                    est_detail_pd.to_csv(f'./log_zoo/singular_val_estimation/cifar10/{net_name}_epoch{args.trained_epoch}.csv', mode='a', header=header)


                else:
                    ## compute the jocabian matrix  
                    svd_topk_list = []
                    for k, encoder in enumerate(encoders): # feature map for each encoder
                            job = torch.autograd.functional.jacobian(encoder, features_list[k])
                            job_dimension = features_list[k][0].shape[0]*features_list[k][0].shape[1]
                            job = job.reshape([job_dimension,job_dimension])

                            # calculate the singular value 
                            svd_topk_list.append(max_svd_compute(job, top_k = 5).tolist())

                    svd_topk_tensor = torch.tensor(svd_topk_list, device=DEVICE)
                    
                    # save the detailed result
                    svd_dict = {}
                    for n in range(len(svd_topk_list)):
                        for m in range(len(svd_topk_list[n])):
                            svd_dict[f'layer{n+1}-top{m+1}'] = [svd_topk_list[n][m]]          

                    svd_pd = pd.DataFrame(svd_dict)

                    if  os.path.exists(f'./log_zoo/singular_val_detail/cifar10/{net_name}_epoch{args.trained_epoch}.csv') or i > 0: 
                        header = None
                    else: 
                        header = True

                    svd_pd.to_csv(f'./log_zoo/singular_val_detail/cifar10/{net_name}_epoch{args.trained_epoch}.csv', mode='a', header=header)

                # save the overall result  
                data_dict = {              
                            'data_set':[args.dataset],
                            'net_name':[net_name],
                            'epoch':[args.trained_epoch],
                            }
                if args.estimation:
                    data_dict['Frobenius_norm'] = [torch.tensor(frobenius_norm_list).max().item()]
                    data_dict['L_infty'] = [torch.tensor(L_infty_norm_list).max().item()]
                    data_dict['L_1'] = [torch.tensor(L_1_norm_list).max().item()]
                else: 
                    data_dict['max_sigma'] = [svd_topk_tensor.max().item()]

                data_pd = pd.DataFrame(data_dict, index=None)
                
                if args.estimation:
                    anotation = 'est'
                else:
                    anotation = 'ana'
                
                if os.path.exists(f'./log_zoo/singular_val_{anotation}.csv') or i > 0: 
                    header = None
                else: 
                    header = True

                data_pd.to_csv(f'./log_zoo/singular_val_{anotation}.csv', mode='a', header=header)

            break


if __name__ == '__main__':
    main()



