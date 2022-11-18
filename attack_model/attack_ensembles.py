import torch
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import save_image
import pandas as pd 
import os 
import math
from tqdm import tqdm
from pathlib import Path

from attack_model.util import * 

class Attack_Ensembles():
    def __init__(self, named_dataset, named_network, named_attack, Lp_for_dist, num_saving, loss_fn, device):

        self.dataset_name, self.dataset = named_dataset
        self.net_name, self.net = named_network
        self.atk_name, self.atk = named_attack
        self.Lp_for_dist = Lp_for_dist
        self.loss_fn = loss_fn
        self.device = device

        self.num_saving = num_saving # the number of saving data
        
        # initalized when attack 
        self.num_saving_per_batch = None 
        self.batch_index = None
        self.log = dict() 


    def save_imgs(self, adv_imgs, ground_truth, preds):
        if self.num_saving > 0:
            index = torch.randint(0, len(adv_imgs), (self.num_saving_per_batch,)) # randomly choose numbers 
            adv_preds = self.net(adv_imgs).argmax(dim=1)
            for i in range(self.num_saving_per_batch):
                folder_path = f'./attack_model/adv_examples/{self.net_name}/{self.atk_name}/'
                img_index = self.batch_index * self.num_saving_per_batch + i
                img_name = f'{self.dataset_name}({img_index})_{ground_truth[index[i]]}->{preds[index[i]]}->{adv_preds[index[i]]}.png'

                if not os.path.exists(folder_path):
                    Path(folder_path).mkdir(parents=True, exist_ok=True)
                    save_image(adv_imgs[index[i]], folder_path + img_name)
                else: 
                    save_image(adv_imgs[index[i]], folder_path + img_name)                    
            self.num_saving -= self.num_saving_per_batch
        else: 
            pass 
    def save_log(self):
        folder_path =  './attack_model/log/'
        file_name = 'attack_results.csv' 
        data_pd = pd.DataFrame(self.log)
        data_pd.loc[:, 'dataset'] = self.dataset_name
        data_pd.loc[:, 'attacks'] = self.atk_name
        data_pd.loc[:, 'network'] = self.net_name

        if not os.path.exists(folder_path + file_name):
            Path(folder_path).mkdir(parents=True, exist_ok=True)                        
            data_pd.to_csv(folder_path + file_name, mode='a', header=True)
        else: 
            data_pd.to_csv(folder_path + file_name, mode='a', header=False)
 
    def _attack(self, dataloader):
        self.batch_index = 0
        self.num_saving_per_batch = math.ceil(self.num_saving/len(dataloader))

        tot_loss = 0
        tot_corrects = 0

        num_of_batches = 0
        num_of_datas = 0

        dist_dict = dict()
        for Lp_norm in self.Lp_for_dist:         
            dist_dict[Lp_norm] = {'mean': [], 'std':[]}
        dist_dict['cor_term'] = []

        for imgs, labels in tqdm(dataloader):

            imgs, labels = imgs.to(self.device), labels.to(self.device)
            adv_imgs = self.atk(imgs, labels).to(self.device) # implement the attack 
            
            # for robust loss  
            tot_loss += self.loss_fn(self.net(adv_imgs), labels).detach().item() # total loss  

            # for robust acc 
            tot_corrects += self.net(adv_imgs).argmax(dim=1).eq(labels).sum().detach().item() # total corrects 

            # for the distance
            for Lp_norm in self.Lp_for_dist: 
                dists = dist_of(imgs, adv_imgs, Lp_norm)
                dist_dict[Lp_norm]['mean'].append(dists.mean())
                dist_dict[Lp_norm]['std'].append(dists.std())
            dist_dict['cor_term'].append(len(adv_imgs)/len(dataloader.dataset))

            # save a fraction of perturbed images  
            self.save_imgs(adv_imgs, labels, self.net(imgs).argmax(dim=1))
            self.batch_index += 1 # this is for naming the saved image 

            num_of_batches += 1 
            num_of_datas += len(adv_imgs) # more accurately collect the numbers of the data


        # average loss and acc
        self.log['rob_loss'] = [tot_loss/num_of_batches]
        self.log['rob_acc'] = [tot_corrects/num_of_datas]

        for Lp_norm in self.Lp_for_dist:
            self.log[Lp_norm + '_dist_mean'] = [(torch.tensor(dist_dict['cor_term'])@torch.tensor(dist_dict[Lp_norm]['mean'])).item()]
            self.log[Lp_norm + '_dist_std'] = [(torch.tensor(dist_dict['cor_term'])@torch.tensor(dist_dict[Lp_norm]['std'])).item()] 

        self.save_log()

    def attack(self, batch_size, frac_size):
        
        _, sub_dataset = random_split(self.dataset, [len(self.dataset) - frac_size, frac_size])
        dataloader = DataLoader(sub_dataset,
                        batch_size=batch_size,
                        num_workers=4,
                        pin_memory=True,
                        shuffle=False)         
        self._attack(dataloader)

 

