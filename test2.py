import matplotlib.pyplot as plt
import torch 
import pandas as pd

from model_zoo.configs import *
from model_zoo.vit_homemade import ViT
from model_zoo.covit import CoViT

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
vit_homemade_conf_dict['vit_D8_E512_H4_R224_P16'] = vit_D8_E512_H4_R224_P16
#vit_homemade_conf_dict['vit_D8_E512_H4_R224_P32'] = vit_D8_E512_H4_R224_P32
#vit_homemade_conf_dict['vit_D12_E512_H8_R224_P32'] = vit_D12_E512_H8_R224_P32


#covit_conf_dict['covit_D1_E512_K3_R224_P16'] = covit_D1_E512_K3_R224_P16
#covit_conf_dict['covit_D1_E512_K3333_R224_P16'] = covit_D1_E512_K3333_R224_P16
#covit_conf_dict['covit_D4_E512_K3_R224_P16'] = covit_D4_E512_K3_R224_P16
#covit_conf_dict['covit_D4_E512_K3333_R224_P16'] = covit_D4_E512_K3333_R224_P16
#covit_conf_dict['covit_D4_E512_K7777_R224_P16'] = covit_D4_E512_K7777_R224_P16
#covit_conf_dict['covit_D8_E512_K3_R224_P16'] = covit_D8_E512_K3_R224_P16
#covit_conf_dict['covit_D8_E512_K1357_R224_P16'] = covit_D8_E512_K1357_R224_P16
covit_conf_dict['covit_D8_E512_K3333_R224_P16'] = covit_D8_E512_K3333_R224_P16
#covit_conf_dict['covit_D8_E512_K7777_R224_P16'] = covit_D8_E512_K7777_R224_P16
#covit_conf_dict['covit_D8_E512_K3333_R224_P32'] = covit_D8_E512_K3333_R224_P32
#covit_conf_dict['covit_D12_E512_4xK3_4xK5_R224_P32'] = covit_D12_E512_4xK3_4xK5_R224_P32
#covit_conf_dict['covit_D16_E512_4xK3_4xK5_R224_P32'] = covit_D16_E512_4xK3_4xK5_R224_P32


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
net_dict = {}
net_dict.update(vit_homemade_dict)
net_dict.update(covit_dict)



df_elder = pd.DataFrame()
for net_name, net in net_dict.items():
    df_tot = pd.read_csv(f'./log_zoo/singular_val_estimation/cifar10/{net_name}_epoch{150}.csv')
    df_old = pd.DataFrame()
    for i in range(1,((df_tot.shape[1]-1)//3 + 1)):
        df_new = pd.DataFrame(df_tot, columns=[f'F-layer{i}', f'L_infty-layer{i}', f'L_1-layer{i}'])
        df_new = df_new.rename(columns={f'F-layer{i}':'F', f'L_infty-layer{i}':'L_infty', f'L_1-layer{i}':'L_1'})
        tmp = (df_new['L_infty']*df_new['L_1'])**(0.5)
        df_new.insert(3, 'estimation', tmp)
        df_new.insert(0, 'layer', f'layer{i}')  
        df_old = pd.concat([df_old, df_new], ignore_index=True)

    df_old.insert(0, 'net_name', net_name)
    df_elder = pd.concat([df_elder, df_old], ignore_index=True)


import seaborn as sns
sns.set_theme(style="whitegrid")

ax = sns.violinplot(x="layer", y="F", hue="net_name",
                    data=df_elder, palette="Set2", split=True,
                    scale="count")
