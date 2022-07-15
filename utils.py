import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import save_image
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import pandas as pd 
import os 

import warnings
from typing import List

from torch import nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math

from model_zoo.vit_homemade import ViT
from model_zoo.covit import CoViT
from model_zoo.configs import *


class Save_dataset:
    def __init__(self, root):
        '''
        root: './dataset/adv_datasets/cifar10_testset_adv_Linf/net_{net_name}_{epoch}/pgd_eps_{eps:.2f}_alpha_{alpha:.2f}_steps_{steps})'
        '''

        if not os.path.exists(root):
            os.makedirs(root)
        
        self.root = root               
        self.mapping = {'img_names':[], 'labels':[]}

    def save_to_dataset(self, named_img):
        '''
        save the create image to a folder one by one instead of a batch of images 
        the folder has 2 files include lables of txt file and mapping of json file
        the folder also include a folder of created images 
        Args: 
            named_img is as triple of (img_name, img, label)
        Output:
            the saved image folder: img/
            lables: ground_truth.txt
            mapping: mapping.json
        '''

        img_name, img, label = named_img
        img_folder = self.root + 'img/'

        if not os.path.exists(img_folder):                           
            os.makedirs(img_folder)

        save_image(img, img_folder + f'{img_name}.png')

        with open(self.root+'ground_truth.txt', 'a') as f:
            f.write(f'{label}\n')

        self.mapping['img_names'].append(img_name)
        self.mapping['labels'].append(label.item())

    def create_json_file(self):
        mapping_dumps = json.dumps(self.mapping)
        with open(self.root+'mapping.json', 'w') as f:
            f.write(mapping_dumps)

class Load_dataset(Dataset):
  def __init__(self, adv_folder, transform=None):
    '''
    This is similar the dataset loader for example datasets.CIFAR10(...) but simple
    The purpose is to load the perturbed images 
    Args:
        adv_folder: where to store the dataset 
        transform: to transform the data set 
    '''
    adv_folder = adv_folder 
    self.transform = transform

    img_folder = adv_folder + 'img/'
    mapping_path = adv_folder + 'mapping.json'

    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    assert len(mapping['img_names'])==len(mapping['labels'])

    num_data = len(mapping['img_names'])

    self.img_names = []
    self.labels = []
    self.img_folder = img_folder
    for i in range(num_data): 
        # why not using self.img_names = mapping['img_names'], 
        # I think because assignment is only a pointer operator
        self.img_names.append(mapping['img_names'][i])
        self.labels.append(mapping['labels'][i])

  def __len__(self):
    return len(self.img_names)
 
  def __getitem__(self, index):
    img_path = self.img_folder + self.img_names[index] + '.png'
    label = self.labels[index]

    img = plt.imread(img_path)
    if self.transform is not None:
        img = self.transform(img)

    return img, label
 
def model_decompose(model, model_def):
    ''' 
        This function is designed for decompose the model vit and CoViT  
        Arg: model is the trained network instance 
             model_def refers to the path of the model where it defines 
    '''

    for mod in model.children():
        if isinstance(mod, model_def.Embedding):
            embedding = mod
        if isinstance(mod, model_def.Encoder):
            encoders = []
            for sub_mod in mod.children():
                if model_def.__name__ == 'model_zoo.vit_homemade':
                    if isinstance(sub_mod, model_def.EncoderBlock):
                        encoders.append(sub_mod) 
                elif model_def.__name__ == 'model_zoo.covit':
                    if isinstance(sub_mod, model_def.EncoderBlock_Conv1D):
                        encoders.append(sub_mod)
        if isinstance(mod, model_def.ClassificationHead):
            head = mod

    return embedding, encoders, head

def plot_result(net_name,load_path, save_path):
        '''
        ploting the training result
        '''
        re = pd.read_csv(load_path, index_col=0)

        fig, ax1 = plt.subplots()
        
        fig.suptitle(net_name)

        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Acc')
        line1 = ax1.plot(re['epoch'], re['train_acc'], label='Train Acc')
        line2 = ax1.plot(re['epoch'], re['val_acc'], label='Test Acc')
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()  

        ax2.set_ylabel('Loss')  
        line3 = ax2.plot(re['epoch'], re['train_loss'], dashes=[6, 2], label='Train Loss')
        line4 = ax2.plot(re['epoch'], re['val_loss'], dashes=[6, 2], label='Test Loss')
        ax2.tick_params(axis='y')

        lines = line1 + line2 + line3 + line4
        labs = [l.get_label() for l in lines]
        ax1.legend(lines, labs, loc=2)

        fig.tight_layout()  
        plt.savefig(save_path)
    

def distance_calc(delta_imgs, Lp_norm):
    '''
    Args: 
    delta_imgs of size (B C H W)
    Lp_norm is the norm used
    '''
    delta_imgs = delta_imgs.flatten(start_dim=1).abs()
    if Lp_norm == 'Linf':
        dists = delta_imgs.max(dim=1)[0]
    if Lp_norm == 'L2':
        dists = (delta_imgs**2).sum(dim=1).sqrt()
    
    return dists


def asr_fn(dataloader):
    pass




def get_loader(args):
    '''
    This function is used to load the small dataset, i.e., cifar 10 and cifar 100 and SVHN
    '''

    transform_train=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.Resize(args.transform_resize),
    transforms.ToTensor(), 
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
    ])


    transform_test=transforms.Compose([
    transforms.Resize(args.transform_resize),
    transforms.ToTensor(), 
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
    ])


#    transform_train = transforms.Compose([
#        transforms.ToTensor(),
#        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#    ])

#    transform_test = transforms.Compose([
#        transforms.ToTensor(),
#        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#    ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./dataset",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./dataset",
                                   train=False,
                                   download=True,
                                   transform=transform_test) 

    elif args.dataset == "cifar100":
        trainset = datasets.CIFAR100(root="./dataset",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./dataset",
                                    train=False,
                                    download=True,
                                    transform=transform_test) 
    elif args.dataset == "svhn":
        trainset = datasets.SVHN(root="./dataset",
                                     train=True,
                                     download=True,
                                     transform=transform_train)

        testset = datasets.SVHN(root="./dataset",
                                    train=False,
                                    download=True,
                                    transform=transform_test) 

    train_loader = DataLoader(trainset,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True,
                             shuffle=False) 

    return train_loader, test_loader





def get_network(args):
    '''
    unfinished
    '''
    if args.net_type == "vit":
        conf = args.net_name
        ViT(
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
    elif args.net_type == "covit":
        pass

    

class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Sets the learning rate of each parameter group to follow a linear warmup schedule between warmup_start_lr
    and base_lr followed by a cosine annealing schedule between base_lr and eta_min.
    .. warning::
        It is recommended to call :func:`.step()` for :class:`LinearWarmupCosineAnnealingLR`
        after each iteration as calling it after each epoch will keep the starting lr at
        warmup_start_lr for the first epoch which is 0 in most cases.
    .. warning::
        passing epoch to :func:`.step()` is being deprecated and comes with an EPOCH_DEPRECATION_WARNING.
        It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
        :func:`get_lr()`. Though this does not change the behavior of the scheduler, when passing
        epoch param to :func:`.step()`, the user should call the :func:`.step()` function before calling
        train and validation methods.
    Example:
        >>> layer = nn.Linear(10, 1)
        >>> optimizer = Adam(layer.parameters(), lr=0.02)
        >>> scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)
        >>> #
        >>> # the default case
        >>> for epoch in range(40):
        ...     # train(...)
        ...     # validate(...)
        ...     scheduler.step()
        >>> #
        >>> # passing epoch param case
        >>> for epoch in range(40):
        ...     scheduler.step(epoch)
        ...     # train(...)
        ...     # validate(...)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            / (
                1
                + math.cos(
                    math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]

