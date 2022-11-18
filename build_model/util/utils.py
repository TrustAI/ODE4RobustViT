import torch
from torchvision import transforms, datasets

from build_model.model_zoo import *
from build_model.optimizers import *


def get_dataset(args):

    transform_train=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.Resize(args.transform_resize),
    transforms.ToTensor(), 
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])


    transform_test=transforms.Compose([
    transforms.Resize(args.transform_resize),
    transforms.ToTensor(), 
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    saving_path = './build_model/dataset'

    if args.dataset == "cifar10":
        train_set = datasets.CIFAR10(root=saving_path,
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        test_set = datasets.CIFAR10(root=saving_path,
                                   train=False,
                                   download=True,
                                   transform=transform_test) 

    elif args.dataset == "cifar100":
        train_set = datasets.CIFAR100(root=saving_path,
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        test_set = datasets.CIFAR100(root=saving_path,
                                    train=False,
                                    download=True,
                                    transform=transform_test) 
    elif args.dataset == "svhn":
        train_set = datasets.SVHN(root=saving_path,
                                     train=True,
                                     download=True,
                                     transform=transform_train)

        test_set = datasets.SVHN(root=saving_path,
                                    train=False,
                                    download=True,
                                    transform=transform_test) 

    return train_set, test_set

def get_network(args):
    if args.net_name == "vit":
        # the dict conf is for debugging 
        conf={ # D: Depth, E: Embedding, H:Head, R:Resolution, P:Patch
            'embedding':{
                            'in_channels': args.in_channels,
                            'img_size': args.img_size,
                            'patch_size': args.patch_size,
                            'em_size': args.em_size
        }, 
                'encoder':{
                            'depth': args.depth,
                            'd_K': args.d_K,
                            'd_V':args.d_V, 
                            'num_heads': args.num_heads, # d_K = heads * d_k
                            'att_drop_out': args.att_drop_out,

                            'MLP_expansion': args.MLP_expansion,
                            'MLP_drop_out': args.MLP_drop_out 

        },
                'cls_head':{
                            'n_classes': args.n_classes
        }
        }

        net = ViT(
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

        # name the net work: e.g., vit_D4_E512_H1_P16
        net_name = f'{args.net_name}_D{args.depth}_E{args.em_size}_H{args.num_heads}_P{args.patch_size[0]}'

    elif args.net_name == "covit":
        conf = { # D: Depth, E: Embedding, K:Kernel size, R:Resolution, P:Patch size
        'embedding':{
                        'in_channels': args.in_channels,
                        'img_size': args.img_size,
                        'patch_size': args.patch_size,
                        'em_size': args.em_size
        }, 
            'encoder':{
                        'depth': args.depth,
                        'kernel_size_group': args.kernel_size_group, # for Conv1D 
                        'stride_group': args.stride_group,
                        'padding_group': args.padding_group,

                        'MLP_expansion': args.MLP_expansion,
                        'MLP_drop_out': args.MLP_drop_out 

        },
            'cls_head':{
                        'n_classes': args.n_classes
        }
        }

        net = CoViT(
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

        # name the net work: e.g., covit_D1_E512_K3_P16
        kernel_size_group = [str(i) for i in args.kernel_size_group]
        kernel_size_group = ''.join(kernel_size_group)
        net_name = f'{args.net_name}_D{args.depth}_E{args.em_size}_H{kernel_size_group}_P{args.patch_size[0]}'

    return net_name, net 


def get_opt(args, net):
    if args.opt_name == 'sam':
        base_optimizer =torch.optim.SGD
        optimizer = SAM(net.parameters(), base_optimizer, lr=args.lr, momentum=args.momentum)    
    return optimizer

def get_lr_scheduler(args, opt, **kwargs):
    lr_sche = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=int(args.train_end_epoch - args.train_start_epoch) + 1, **kwargs)
    return lr_sche





