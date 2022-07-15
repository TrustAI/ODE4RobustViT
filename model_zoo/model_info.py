from tabnanny import verbose
from configs import *
from vit_homemade import ViT
from covit import CoViT
from torchsummary import summary

vit_homemade_conf_dict = {}
covit_conf_dict = {}
vit_homemade_dict = {}
covit_dict = {}

vit_homemade_conf_dict['vit_D1_E512_H1_R224_P16'] = vit_D1_E512_H1_R224_P16
vit_homemade_conf_dict['vit_D1_E512_H4_R224_P16'] = vit_D1_E512_H4_R224_P16
vit_homemade_conf_dict['vit_D4_E512_H1_R224_P16'] = vit_D4_E512_H1_R224_P16
vit_homemade_conf_dict['vit_D4_E512_H4_R224_P16'] = vit_D4_E512_H4_R224_P16
vit_homemade_conf_dict['vit_D8_E512_H1_R224_P16'] = vit_D8_E512_H1_R224_P16
vit_homemade_conf_dict['vit_D8_E512_H4_R224_P16'] = vit_D8_E512_H4_R224_P16
vit_homemade_conf_dict['vit_D8_E512_H4_R224_P32'] = vit_D8_E512_H4_R224_P32
vit_homemade_conf_dict['vit_D12_E512_H8_R224_P32'] = vit_D12_E512_H8_R224_P32

covit_conf_dict['covit_D1_E512_K3_R224_P16'] = covit_D1_E512_K3_R224_P16
covit_conf_dict['covit_D1_E512_K3333_R224_P16'] = covit_D1_E512_K3333_R224_P16
covit_conf_dict['covit_D4_E512_K3_R224_P16'] = covit_D4_E512_K3_R224_P16
covit_conf_dict['covit_D4_E512_K3333_R224_P16'] = covit_D4_E512_K3333_R224_P16
covit_conf_dict['covit_D4_E512_K7777_R224_P16'] = covit_D4_E512_K7777_R224_P16
covit_conf_dict['covit_D8_E512_K3_R224_P16'] = covit_D8_E512_K3_R224_P16
covit_conf_dict['covit_D8_E512_K1357_R224_P16'] = covit_D8_E512_K1357_R224_P16
covit_conf_dict['covit_D8_E512_K3333_R224_P16'] = covit_D8_E512_K3333_R224_P16
covit_conf_dict['covit_D8_E512_K7777_R224_P16'] = covit_D8_E512_K7777_R224_P16
covit_conf_dict['covit_D8_E512_K3333_R224_P32'] = covit_D8_E512_K3333_R224_P32
covit_conf_dict['covit_D12_E512_4xK3_4xK5_R224_P32'] = covit_D12_E512_4xK3_4xK5_R224_P32
covit_conf_dict['covit_D16_E512_4xK3_4xK5_R224_P32'] = covit_D16_E512_4xK3_4xK5_R224_P32

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


if __name__ == '__main__':

    for key, value in covit_dict.items():
        input("Press Enter to continue...")
        print('#'*50)
        print(key)
        summary(value, (3, 224, 224), device='cpu')
        

   

