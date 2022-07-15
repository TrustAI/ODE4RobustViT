import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.dataloader import DataLoader

from tqdm import tqdm
import logging


######
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
net_name = 'mresnet_Conv1d_D1_E96_K1357_R224_P16'
from mresnet_conv1d import mresnet_Conv1d_D1_E96_K1357_R224_P16 # do not forget to set ctf of network
net = mresnet_Conv1d_D1_E96_K1357_R224_P16
######

# hyper-parameters 
EPOCHS = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASS = 10
BATCH_SIZE = 512
RESIZE = [224,224]
lr = 0.001
#ATT_BATCH_SIZE = 256

# log initilization
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s   %(levelname)s   %(message)s')
logger = logging.getLogger(net_name)

# log file 
handler_file = logging.FileHandler(net_name, 'w')
formatter = logging.Formatter('%(asctime)s  %(message)s')
handler_file.setFormatter(formatter) 
logger.addHandler(handler_file)plt.imshow(img.cpu())

train_CIFAR10 = datasets.CIFAR10(
    root='./dataset',
    train=True,
    download=True,
    transform=transforms.Compose([
    transforms.Resize(RESIZE),
    transforms.ToTensor(), 
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
    ])
)

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


train_loader = DataLoader(
    dataset=train_CIFAR10, batch_size=BATCH_SIZE, shuffle=True
    )
test_loader = DataLoader(
    dataset=test_CIFAR10, batch_size=BATCH_SIZE
)

# set optimizer and neural network
optimizer = torch.optim.Adam(net.parameters() ,lr=lr)
net = net.to(DEVICE)

# visualization 
#viz = Visdom(port=8090)

# training 
test_acc_old = 0
for epoch in range(EPOCHS):
    
    train_total_loss = 0
    train_total_corrects = 0
    
    net.train()
    for imgs, labels in tqdm(train_loader):

        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = net(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        logits = net(imgs)
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=1)
        train_total_loss += loss.detach().item() 
        train_total_corrects += preds.eq(labels).sum().detach().item()

    # train loss and acc
    train_loss = train_total_loss/len(train_loader)
    train_acc = train_total_corrects/len(train_loader.dataset)

    # clean test loss and acc
    test_loss, test_acc = test(dataloader=test_loader, model=net, loss_fn=F.cross_entropy, device=DEVICE)
    print(f'epoch: {epoch} train_loss: {train_loss} train_acc: {train_acc} test_loss: {test_loss} test_acc: {test_acc}')

    # robust loss and acc
#    attack_loader = DataLoader(
#    dataset=test_CIFAR10, batch_size=ATT_BATCH_SIZE, shuffle=True
#)   
#    X_hat, Z_hat, y_1, y_2, X = attacking(attack_loader, net, device=DEVICE, saving=False, attack_method='fgsm', epsilon=0.01)
#    robust_loss, robust_acc = acc2(X_hat, y_1, net, DEVICE, target_class='all')

    # loss for X not-misclassified
#    loss_for_corrects = F.cross_entropy(net(X), y_1)/len(X)
#    loss_for_corrects = loss_for_corrects.detach().item()

    # experiment log
    logger.info(f'epoch: {epoch} train_loss: {train_loss} train_acc: {train_acc} test_loss: {test_loss} test_acc: {test_acc}') 
                  

    # visualization  
#    viz.line([[train_loss, test_loss]], \
#    [epoch], win=f'{net_name}_loss', update='append', opts=dict(title='Loss',legend=['Train', 'Test' ]))
    
#    viz.line([[train_acc, test_acc,]], \
#    [epoch], win=f'{net_name}_acc', update='append', opts=dict(title='Acc',legend=['Train', 'Test']))
    if epoch % 20 == 0:
        torch.save(net.state_dict(),f'./trained_networks/{net_name}_epoch{epoch}.pth')




import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.dataloader import DataLoader
import vit
from utils import eigenvals
from attacks import attacking
import torch 
from utils import model_decompose


BATCH_SIZE = 10
RESIZE = [224,224]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

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

def eigen_ana(data_loader, name_network_category, epoch=180, device = DEVICE, attacking_method='fgsm', **kwargs):
    '''
    Argument:data_loader is the data loader to be attacked, usually the test_loader
                named_network is the tuple of (network name, network)
                device is the of cpu and gpu 
                attacking method is of 'fgsm', 'pgd', 'cw', ...
                **kwargs is the key word arguments for the attacking method, for fgsm ti is epsilon = 0.005
    '''


    net_name, net, cate = name_network_category
    net.load_state_dict(torch.load(f'./ckp/{net_name}_epoch{epoch}.pth'))


    # X_hat is the perturbed correctly classified imgs
    # Z_hat is the misclassificed imgs 
    # y_correct is the correctly classified labels 
    # y_failed is the misclassified ones 
    # X_correct is the corretly classified imgs 
    X_attacked, X_failed, y_correct, y_failed, X_org = attacking(data_loader=data_loader, network = net, \
                                                        device=device, saving=False, \
                                                        attack_method=attacking_method, **kwargs)

    def _eigen_ana(named_imgs):
        '''
        Arg:named_X is of (name, X), where X refers to the input image which is the result of attacking above 
            other variables to be used includes embedding, encoders, and function of eigenvals 
        Out:writting the maximum of real part of the eigenvals calculated to the file 
        '''

        name, imgs = named_imgs
        X = []
        for i,img in enumerate(imgs):
            img = img[None,:,:,:] 
            with torch.no_grad():
                X.append(embedding(img))

            eigenvals_maxreal= []
            for j,encoder in enumerate(encoders):
                X.append(encoder(X[j]))
                eigenvals_maxreal.append(eigenvals(encoder=encoder, X = X[j], device=DEVICE))
                
            with open('./log/eigen_maxreal.txt', 'a') as f:
                f.write(f'\n{name}[{i}]:')
                for eigen in eigenvals_maxreal:
                    f.write(f' {eigen}')


    # Decompose the model w.r.t. their corresponding parts 
    embedding, encoders, head = model_decompose(net, cate)
    _eigen_ana((f'{net_name}_original', X_org))
    _eigen_ana((f'{net_name}_attacked_{attacking_method}', X_attacked))

from mresnet_conv1d import mresnet_Conv1d_D8_E96_K3333_R224_P16
import mresnet_conv1d
from vit import vit_D8_E96_H4_R224_P16
import vit
from collections import OrderedDict

net_store = OrderedDict()
net_store['vit'] = [vit, 'vit_D8_E96_H4_R224_P16', vit_D8_E96_H4_R224_P16]
net_store['mresnet_Conv1d'] = [mresnet_conv1d, 'mresnet_Conv1d_D8_E96_K3_R224_P16', mresnet_Conv1d_D8_E96_K3333_R224_P16]

#for store in net_store: 
#    eigen_ana(data_loader=test_loader, name_network_category=(store[1], store[2], store[0]), \
#          epsilon=0.005)



import torch.nn.functional as F
import torchvision.transforms as transforms
import torch
import torch.nn as nn 
from torch.utils import data
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm
import pandas as pd 


def acc(dataloader, model, device = None, target_class = 'all'):
    size = len(dataloader.dataset)
    num_batch = len(dataloader)
    if target_class != 'all':
         size_target_class = sum([i == target_class for i in dataloader.dataset.targets]) 
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        # This is very important
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            if target_class == 'all':
                pred = model(X)
                test_loss += nn.CrossEntropyLoss()(pred, y).detach().item() 
                correct += (pred.argmax(1) == y).type(torch.float).sum().detach().item()
            else: 
                idx = y == target_class
                pred = model(X[idx])
                test_loss += nn.CrossEntropyLoss()(pred, y[idx]).detach().item()
                correct += (pred.argmax(1) == y[idx]).type(torch.float).sum().detach().item()

    if target_class == 'all':
        test_loss /= num_batch
        correct /= size
    else:
        test_loss /= num_batch
        correct /= size_target_class
    return test_loss, correct



def acc2(X, y, network, device = None, target_class = 'all'):
    network = network.to(device)
    X = X.to(device)
    y = y.to(device)

    network.eval()
    with torch.no_grad():
        if target_class == 'all':
            size = len(X)
            logits = network(X)
            loss = F.cross_entropy(logits, y).detach().item()

            preds = logits.argmax(dim=1)
            corrects = preds.eq(y).sum().detach().item()
        else:
            idx = y == target_class
            size = len(X[idx])
            logits = network(X[idx])
            loss = F.cross_entropy(logits, y[idx]).detach().item()

            preds = logits.argmax(dim=1)
            corrects = preds.eq(y[idx]).sum().detach().item()

    return loss, corrects/size

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss, correct = 0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= num_batches
    correct /= size

    return train_loss, correct



def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size

    return test_loss, correct

def read_log_train(filename):
    with open(filename, 'r') as f:
        epochs = []
        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []
        while True:
            lines = f.readline()
    
            if lines == '': break
    
            info = lines.split()
            epochs.append(float(info[3]))
            train_loss.append(float(info[5]))
            train_acc.append(float(info[7]))
            test_loss.append(float(info[9]))
            test_acc.append(float(info[11]))
    
    return epochs, train_loss, train_acc, test_loss, test_acc

def plot_img(net_namelist = None, indicator = 'Acc',  
             layout = (1, 1), num_ticks_x=10, num_ticks_y=10):

    plt.figure(figsize=(5*layout[1], 5*layout[0]))
    cnt = 0

    for i in range(layout[0]):
        for j in range(layout[1]):
            cnt += 1 

            # read the log data 
            # epochs, acc and loss data are all store in list
            filename = f'./log/{net_namelist[cnt-1]}'
            epochs, train_loss, train_acc, test_loss, test_acc = read_log_train(filename)

            if indicator == 'Acc':
                train_data = train_acc
                test_data = test_acc
                y_max = 1

            elif indicator == 'Loss':
                train_data = train_loss
                test_data = test_loss
                y_max = max(train_data + test_data)*1.2

            step_x = len(epochs)/num_ticks_x
            step_y = y_max/num_ticks_y
            
            plt.subplot(layout[0],layout[1],cnt)
            plt.xticks(np.arange(0, len(epochs), step=step_x))
            plt.yticks(np.arange(0, y_max, step=step_y))
            plt.ylim(0, 1)

            plt.title(net_namelist[cnt-1])
            if i == layout[0]-1:
                plt.xlabel("Epochs")
            if j == 0: 
                plt.ylabel(indicator)
            
            plt.plot(epochs, train_data, label=f'Train {indicator}')
            plt.plot(epochs, test_data, label=f'Test {indicator}')

            plt.legend(loc='lower right')

    plt.show()

def read_log_attack(filename):
    with open(filename, 'r') as f:
        epochs = []
        epsilon = []
        robust_loss = []
        asr = []
        while True:
            lines = f.readline()
    
            if lines == '': break
    
            info = lines.split()
            epochs.append(float(info[3]))
            epsilon.append(float(info[5]))
            robust_loss.append(float(info[6][12:]))
            asr.append(float(info[7][4:]))
    return epochs, epsilon, robust_loss, asr

def plot_img_fgsm(net_namelist = None, indicator = 'Loss',  
             layout = (1, 1), num_ticks_y=10, loc = 'lower right'):

    plt.figure(figsize=(5*layout[1], 5*layout[0]))
    cnt = 0

    for i in range(layout[0]):
        for j in range(layout[1]):
            cnt += 1 

            # read the log data 
            # epochs, acc and loss data are all store in list
            filename = f'./log/{net_namelist[cnt-1]}'
            epochs, epsilon, robust_loss, asr = read_log_attack(filename)
            eps = list(set(epsilon))
            eps.sort()

            epo = list(set(epochs))
            epo.sort()

            if indicator == 'ASR':
                attack_data = asr
                y_max = 1

            elif indicator == 'Loss':
                attack_data = robust_loss
                y_max = max(robust_loss)*1.2

            step_y = y_max/num_ticks_y
            
            plt.subplot(layout[0],layout[1],cnt)
            plt.xticks(epo)
            plt.yticks(np.arange(0, y_max, step=step_y))

            plt.title(net_namelist[cnt-1])
            
            if i == layout[0]-1:
                plt.xlabel("Epochs")
            if j == 0: 
                plt.ylabel(indicator)

            for e in eps:
                idx = np.array(epsilon) == np.array(e)
                epochs = np.array(epochs)
                attack_data = np.array(attack_data)
                plt.plot(epochs[idx], attack_data[idx], label=f'Eps = {e}')

            plt.legend(loc=loc)

    plt.show()



def read_log(file_path, sep = ' ', alpha_pos = [0], stat = True, catalog_pos = 0, indent = [0,-4]):
    '''
        This function is to read experiments result which is in txt file 
        and formated line by line and each line indicates one experiment result
        In one line, the exact number or title are separated by sep
        The mumber will be automaticly converted to numeric
        If Stat = Ture, the basic statistic will be also calculated 

        Arg:
        file_path: the path of log 
        sep: the separation 
        alpha_pos: the position of the non-numerical, can be [0 ,3, ....]
        stat: whethe calulate the statistics 
        catalog_pos: the position of catalog, should be only 1,
        indent: the indent for catalog, e.g., 'vit[0]:' after indent 4 -> 'vit'

        Output: 
        a 2-dimension list and the statistics        
    '''

    lines_list = []
    with open(file_path, 'r') as f:
        for line in f.readlines():

            line_list = []
            cnt = 1
            item = ''
            for pos in line:
                cnt += 1    
                if not pos.isspace():
                    item += pos
                if pos == sep or cnt == len(line):
                    line_list.append(item)
                    item = ''

            if not line_list == []:        
                lines_list.append(line_list)

    # convert strings to number
    for i in range(len(lines_list)):
        for j in range(len(lines_list[i])):
            if j not in alpha_pos:
                lines_list[i][j] = float(lines_list[i][j])

    # calculate basic statics for numerical data 
    if stat:
        # construct the statistic 
        net_name_dict = {}
        for line_list in lines_list:
            catalog = line_list[catalog_pos][indent[0]:indent[1]]  
            if catalog not in net_name_dict:
                net_name_dict[catalog] = []        
        
        # filling the numbers to the statistics
        for line_list in lines_list:
            for j in range(len(line_list)):
                if j != catalog_pos:
                    catalog = line_list[catalog_pos][indent[0]:indent[1]]
                    net_name_dict[catalog].append(line_list[j])

        # convert to tensor and calculation of statistics
        statistics = {}
        for key in net_name_dict.keys():
            net_name_dict[key] = torch.tensor(net_name_dict[key]).reshape(-1, len(lines_list[0])-1)
            statistics[key] = [net_name_dict[key].mean(dim=0), net_name_dict[key].std(dim=0)]
    else: statistics = None

    return lines_list, statistics


def finetune(dataloaders, named_pretrained_net, epochs=5, loss_fn=F.cross_entropy, \
             optimizer = None, device = None, verbose = True):
    '''
        This function is to finetune a pretrained model

        Args
        dataloaders: the data to be finetuned, (trainloader, testloader) 
        named_pretrained_net: (network name, network)
        epochs: the total training epochs on the new dataset 
        loss_fn: loss function to train the model 
        optimizer: the optimizer to be used 
        num_class: the number of class for dataset to be finetuned 
        device: cpu or gpu 
        verbose: show the training and testing accuracy 
    '''
    train_loader, test_loader = dataloaders
    net_name, net = named_pretrained_net

    for epoch in range(epochs):
        net.train()
        for imgs, labels in tqdm(train_loader):

            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = net(imgs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        if verbose:
            # train loss and acc
            train_loss, train_acc = test(dataloader=train_loader, model=net, loss_fn=loss_fn, device=device)

            # clean test loss and acc
            test_loss, test_acc = test(dataloader=test_loader, model=net, loss_fn=loss_fn, device=device)
            print(f'epoch: {epoch} train_loss: {train_loss} train_acc: {train_acc} test_loss: {test_loss} test_acc: {test_acc}') 
                    
        #if epoch % 20 == 0:
        torch.save(net.state_dict(), f'./ckp/{net_name}_epoch{epoch}.pth')

def save_data(data_list, label_list, saving_dir, size = 'full'):

    mapping = {'data_path':[], 'label':[]}
    
    if size == 'full':
        size = []
        for data in data_list:
            size.append(len(data))

    # save data_list
    for i in range(len(data_list)):
        for j in range(size[i]):
            save_image(data_list[i][j], f'./{saving_dir}/img00{i}--{j}.png')            
            mapping['data_path'].append(f'./{saving_dir}/img00{i}--{j}.png')
            mapping['label'].append(label_list[i][j].item())

    mapping_dumps = json.dumps(mapping)
    with open(f'./{saving_dir}/mapping.json', 'w') as f:
        f.write(mapping_dumps)


# attack py FGSM
def _fgsm(X, y, model, epsilon):
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

def _pgd(X, y, model, y_target, Lp, epsilon, alpha, iter_max, restarts_num = 1, zero_init = False):
    
    max_loss = torch.zeros(y.shape[0]).to(y.device)
    max_delta = torch.zeros_like(X)

    for i in range(restarts_num):
        if zero_init:
            delta = torch.zeros_like(X, requires_grad=True) 
        else:
            delta = torch.rand_like(X, requires_grad=True)
            delta.data = delta.data * 2 * epsilon - epsilon #[-epsilon, epsilon]

        for j in range(iter_max):
            # different setup of loss function
            if y_target == 'all':
                loss = nn.CrossEntropyLoss()(model(X + delta), y)
            else:
                yp = model(X + delta)
                loss = 2*yp[:,y_target].sum() - yp.sum()
                # maximize loss for target class over all other classes not only the ground truth
                # loss = (yp[:,y_targ] - (yp - yp[:,y_targ])).sum()

            loss.backward()
            
            if Lp == 'inf':
                delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            else: 
                if Lp < 0: raise Exception('Lp should > 0')

                Lq = Lp/(Lp - 1)
                delta.data += alpha*delta.grad.detach().sign() * \
                    (delta.grad.detach().abs() ** Lq / \
                    (delta.grad.detach().abs() ** Lq).flatten(start_dim=1).sum(dim=1)[:,None, None, None])**(1/Lp)
                delta.data = torch.min(torch.max(delta.detach(), -X), 1-X)
                # clip X+delta to [0,1]
                delta.data *= epsilon / \
                ((delta.detach().abs()**Lp).flatten(start_dim=1).sum(dim=1)[:,None, None, None]**(1/Lp)).clamp(min=epsilon)
                # projection to the L_p norm ball 

            delta.grad.zero_()

        all_loss = nn.CrossEntropyLoss(reduction='none')(model(X+delta),y)
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        # each delta in max_delta has the largest loss

        max_loss = torch.max(max_loss, all_loss)
        # torch.max will subtitute to smaller value for each dimension 
    return max_delta


def _attacking(X, y, network, attack_fun=_fgsm, **kwargs):
    network.train() # for back-prapagation 

    delta = attack_fun(X, y, network, **kwargs)
    X_hat = X + delta
    y_p = network(X_hat)
    return y, y_p, X_hat.detach() # y is the ground truth, y_p is the prediction,X_hat is the perturbed data

def attacking(data_loader, network, device=device, saving=False, attack_method = 'fgsm', **kwargs):
    # input is a data_loader and network
    # the output X_hat is the perturbed data set which is correctly predicted
    # Z_hat is the misclassified part
    # y is the corresponding ground turth label

    network = network.to(device)
    # X_hat is the correct predicted img while been perturbed
    for X, y in data_loader:

        X, y = X.to(device), y.to(device)

        logits = network(X)
        preds = logits.argmax(dim=1)
        idx = preds == y

        if attack_method == 'fgsm':
            y_o, y_p, X_hat = _attacking(X[idx], y[idx], network, _fgsm, **kwargs)
        elif attack_method == 'pgd':
            y_o, y_p, X_hat = _attacking(X[idx], y[idx], network, _pgd, **kwargs)
        Z_hat = X[idx==False].detach()

        if saving: # this saving is for dataloader 
            save_data([X_hat, Z_hat], [y[idx], y[idx==False]], 'adv_data')
        else:
            break

    return X_hat, Z_hat, y[idx], y[idx==False], X[idx].detach()

def attack_test(data_loader, network, device=device, save_img = True,
                target_class = 'all', attack_method = 'fgsm', **kwargs):

    num_batch = len(data_loader)
    network = network.to(device)
    loss_tot, corrects_tot = 0, 0 

    for X, y in data_loader:

        X, y = X.to(device), y.to(device)

        logits = network(X)
        preds = logits.argmax(dim=1)
        idx = preds == y

        if attack_method == 'fgsm':
            y_o, y_p, X_hat = _attacking(X[idx], y[idx], network, _fgsm, **kwargs)
        elif attack_method == 'pgd':
            y_o, y_p, X_hat = _attacking(X[idx], y[idx], network, _pgd, **kwargs)
        
        loss, corrects = acc2(X_hat, y[idx], network, device, target_class)
        loss_tot += loss 
        corrects_tot += corrects

    # show the perturbed image 
    if save_img:
        imgs_save(X_hat, y_o, y_p, path=f'./adv_imgs/cifar10_{attack_method}_{target_class}_{kwargs}.png')

    return loss_tot/num_batch, corrects_tot/num_batch


def jacob_mat(x, fun, device_x):
    # x and y are both matrix (torch.tensor([n,m])) 
    # x should be the leaf of the computational graph
    # fun is the function to be autograded 
    # device_x refer to the device for input x

    size_x = x.shape
    x = x.to(device_x)
    x.requires_grad_()
    
    y = fun(x)
    size_y = y.shape
    
    J = torch.tensor([])

    for i in range(size_y[0]):
        for j in range(size_y[1]):
            grad = torch.autograd.grad(inputs=x, outputs=y[i,j], retain_graph=True)[0].reshape([-1,1]).detach().clone()
            J = torch.cat([J, grad], dim=1)

    '''the output J is jocbain of y to x of size (y_n*y_m, x_n*x_m)'''
    return J



# input: img(B, N+1, Em)
# outpuy: img(B, N+1, Em)
class MultiHeadConv1D(nn.Module):
    def __init__(self, em_size: int=512, kernel_size_group: tuple=(1,3,5,7),\
                 stride_group: tuple=(1,1,1,1), padding_group: tuple=(0,1,2,3)
                ):
        super().__init__()
        self.em_size = em_size 
        self.kernel_size_group = kernel_size_group
        self.stride_group = stride_group
        self.padding_group = padding_group

        self.num_heads = len(kernel_size_group)
        self.projection = nn.Linear(em_size, em_size)

    def forward(self, x):
        channels = self.em_size//self.num_heads
        device = x.device
        assert self.num_heads % self.num_heads == 0, "embedding size should be divided by #heads"
        x = torch.permute(x, (0 ,2, 1)) # x is of (B, Em, N+1)
        x = rearrange(x, 'b (h d) n-> h b d n', h = self.num_heads)
        
        for k, s, p, h in zip(self.kernel_size_group, self.stride_group, self.padding_group, range(self.num_heads)):
            conv1d = nn.Conv1d(in_channels = channels, out_channels =channels, 
                                kernel_size=k, stride=s, padding=p)
            conv1d = conv1d.to(device)
            x[h] = conv1d(x[h])
        x = rearrange(x, 'h b d n->b n (h d)')
        out = self.projection(x)
        return out


   

