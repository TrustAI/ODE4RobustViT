import torch
from torch.utils.data.dataloader import DataLoader
import pandas as pd 
import os 
from tqdm import tqdm
from pathlib import Path

from build_model.model_zoo import *
from build_model.optimizers import *
from build_model.util import *



class Learner:
    def __init__(self, named_dataset, named_network, \
                       optimizer, lr_scheduler, loss_fn, device):

        self.dataset_name, self.train_set, self.test_set = named_dataset 
        self.net_name, self.net = named_network
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.device = device

        # those are initialized during training 
        self.start_epoch = None  
        self.end_epoch = None 
        self.batch_size = None 
        self.ckp_freq = None 

    def epoch_run(self, dataloader, backward_pass=False):
        '''
        This function is to do each epoch run of the dataset
        '''
        total_loss = 0
        total_corrects = 0
        
        num_of_batches = 0
        num_of_datas = 0
        for imgs, labels in tqdm(dataloader):
            num_of_batches += 1 
            num_of_datas += len(imgs) # more accurately collect the numbers of the data

            imgs, labels = imgs.to(self.device), labels.to(self.device)

            logits = self.net(imgs) # in logit space 
            loss = self.loss_fn(logits, labels)
            if backward_pass:

                # check whether use sam to optimize the network
                is_sam = self.optimizer.__class__.__name__ == 'SAM' 
                if is_sam:
                    def closure():
                        loss = self.loss_fn(self.net(imgs), labels)
                        loss.backward()
                        return loss

                self.optimizer.zero_grad()
                loss.backward()
                if is_sam:
                    self.optimizer.step(closure)
                else: 
                    self.optimizer.step()

                # check whether using learning scheduler 
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
 
            preds = logits.argmax(dim=1) # the prediction 
            total_loss += loss.detach().item() # total loss 
            total_corrects += preds.eq(labels).sum().detach().item() # total corrects 

        # average loss and acc
        loss = total_loss/num_of_batches
        acc = total_corrects/num_of_datas

        return loss, acc

    def train(self, start_epoch, end_epoch, batch_size, ckp_freq=None, verbose=True):
        
        '''
        Args:
            epochs: is of a integer indicating training from start or a tulpe indicating training from (start, end)
            save_frequency: if is None then not saving 
            verbose: showing the training result  
        '''
        self.start_epoch = start_epoch 
        self.end_epoch = end_epoch
        self.batch_size = batch_size
        self.ckp_freq = ckp_freq

        # the path to store the network 
        
        # loader the train_set 
        train_loader = DataLoader(self.train_set,
                                    batch_size=batch_size,
                                    num_workers=4,
                                    pin_memory=True)

        assert start_epoch >= 1, "start epoch has to start from 1 instead of 0 or below"
        if start_epoch > 1: # if the start_epoch is not zero, we load the network from the start_epoch             
            file_path = f'./build_model/ckp/{self.dataset_name}/{self.net_name}_epoch({start_epoch}).pth'
            assert os.path.exists(file_path), "network not found"
            self.net.load_state_dict(torch.load(file_path))

        self.net.train() # training mode
        for epoch in range(start_epoch, end_epoch+1): # the training will include start and end epoch 
            # the training can start from a perviously trained position of epoch 
            train_loss, train_acc = self.epoch_run(train_loader, backward_pass=True)

            if verbose: # showing the detail of the training 
                print(f'train: epoch {epoch} loss {train_loss:.3f}, acc {train_acc:.3f}')
            if ckp_freq is not None:
                if epoch % ckp_freq == 0: # save the model for give steps                    
                    file_dir = f'./build_model/ckp/{self.dataset_name}/'
                    if not os.path.exists(file_dir):
                        Path(file_dir).mkdir(parents=True, exist_ok=True)                        
                        torch.save(self.net.state_dict(), file_dir+f'{self.net_name}_epoch({epoch}).pth')
                    else:
                        torch.save(self.net.state_dict(), file_dir+f'{self.net_name}_epoch({epoch}).pth')
            
    def evaluate(self, dataloader):
                
        self.net.eval() # evaluation model
        with torch.no_grad():
            loss, acc  = self.epoch_run(dataloader)
        return loss, acc

    def validate(self, start_epoch, end_epoch, batch_size, ckp_freq):

        train_loader = DataLoader(self.train_set,
                                batch_size=batch_size,
                                num_workers=4,
                                pin_memory=True)
        test_loader = DataLoader(self.test_set,
                                batch_size=batch_size,
                                num_workers=4,
                                pin_memory=True,
                                shuffle=False) 
        for epoch in range(start_epoch, end_epoch+1, ckp_freq): # note that the start epoch is the ckp_freq
            # load network from ckp 
            net_path = f'./build_model/ckp/{self.dataset_name}/{self.net_name}_epoch({epoch}).pth'
            assert os.path.exists(net_path), "network not found"
            self.net.load_state_dict(torch.load(net_path)) # load the trained model 

            train_loss, train_acc = self.evaluate(train_loader)
            print(f'training: epoch {epoch} loss {train_loss} acc {train_acc}')
            val_loss, val_acc = self.evaluate(test_loader)
            print(f'val: epoch {epoch} loss {val_loss} acc {val_acc}')
            data_pd = pd.DataFrame(
                                {
                                'dataset': [self.dataset_name],
                                'train_loss':[train_loss],
                                'train_acc':[train_acc],
                                'val_loss':[val_loss],
                                'val_acc':[val_acc]
                                }, index=[epoch])
            file_dir = f'./build_model/log/'
            if not os.path.exists(file_dir + f'{self.net_name}.csv'):
                Path(file_dir).mkdir(parents=True, exist_ok=True)                        
                data_pd.to_csv(file_dir + f'{self.net_name}.csv', mode='a', header=True)
            else: 
                data_pd.to_csv(file_dir + f'{self.net_name}.csv', mode='a', header=False)

# for test 
# def main():

#     parser = argparse.ArgumentParser(description='Training vits and covits')

#     # the class of the model, e.g., vit or covit  
#     parser.add_argument('--net_name', type=str, default='vit', choices=['vit, covit'], 
#                         help='the type of the neural network, e.g., vit or covit')

#     # configuration for ViT 
#     parser.add_argument('--in_channels', type=int, default=3, 
#                         help='the number of input channles, e.g., 3')
#     parser.add_argument('--img_size', nargs='+', type=int, default=[224, 224], 
#                         help='the input image size, e.g., (224, 224)')
#     parser.add_argument('--patch_size', nargs='+', type = int, default=[16, 16], 
#                         help='the size of patches for patch embeddings, e.g., (16, 16)')
#     parser.add_argument('--em_size', type=int, default=128, 
#                         help='the embedding size, e.g., 512')

#     parser.add_argument('--depth', type=int, default=4, 
#                         help='the number basic blocks of vits and covits ')
#     parser.add_argument('--d_K', type=int, default=128, 
#                         help='the dimension of the Key')
#     parser.add_argument('--d_V', type=int, default=128, 
#                         help='the dimension of the Value')
#     parser.add_argument('--num_heads', type=int, default=4, 
#                         help='the number of heads')
#     parser.add_argument('--att_drop_out', type=float, default=0., 
#                         help='the drop_out rate for self attention')
#     parser.add_argument('--MLP_expansion', type=int, default=4, 
#                         help='the expansion rate for MLP layer in transformer encoder')
#     parser.add_argument('--MLP_drop_out', type=float, default=0., 
#                         help='the drop_out rate for MLP layers')

#     parser.add_argument('--n_classes', type=int, default=10, 
#                         help='the number of classes')

#     # configuration for CoViT
#     parser.add_argument('--kernel_size_group', nargs='+', type=int, default=[3,3,3,3], # equivelent to 4 heads 
#                         help='the number of classes')
#     parser.add_argument('--stride_group', nargs='+', type=int, default=[1,1,1,1], 
#                         help='the number of classes')
#     parser.add_argument('--padding_group', nargs='+', type=int, default=[1,1,1,1], 
#                         help='the number of classes')

#     # dataset configuration 
#     parser.add_argument('--dataset', type=str, default='cifar10', 
#                         help='the data set to be trained with')
#     parser.add_argument('--transform_resize', type=int, default=224, 
#                         help='transform the inputs: resize the resolution')
    
#     # training settings 
#     parser.add_argument('--opt_name', type=str, default='sam', 
#                         help='the name for optimizer')
#     parser.add_argument('--lr', type=float, default=0.1, 
#                         help='the learning rate for the opt')
#     parser.add_argument('--momentum', type=float, default=0.9, 
#                         help='the momentum for the opt')                        
#     parser.add_argument('--train_start_epoch', type=int, default=1, 
#                         help='the number of input channles')
#     parser.add_argument('--train_end_epoch', type=int, default=4, 
#                         help='the number of input channles')
#     parser.add_argument('--train_ckp_freq', type=int, default=2, 
#                         help='the frequence for saving network')
#     parser.add_argument('--batch_size', type=int, default=256, 
#                         help='the batch size for training and validating')

#     # arguments for validation 
#     parser.add_argument('--val_start_epoch', type=int, default=2, # notice that start_epoch for validation should be the same with val_ckp_freq
#                         help='the number of input channles')
#     parser.add_argument('--val_end_epoch', type=int, default=5, 
#                         help='the number of input channles')
#     parser.add_argument('--val_ckp_freq', type=int, default=2, # it must be an integer multiple of train_ckp_freq
#                         help='the frequence for saving network')


#     args = parser.parse_args()


#     DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # load the train loader, prepare for training 
#     train_set, test_set = get_dataset(args)
#     net_name, net = get_network(args) # net_name e.g., covit_D1_E512_K3_P16
#     net = net.to(DEVICE)    

#     optimizer = get_opt(args,net)
#     lr_scheduler = get_lr_scheduler(args, optimizer, steps_per_epoch=math.ceil(len(train_set)/args.batch_size)) # step = len(train_loader)

#     # initialization of the learner which is used to train and validate the network 
#     learner = Learner((args.dataset, train_set, test_set), (net_name, net), optimizer, lr_scheduler, F.cross_entropy, device=DEVICE)
#     learner.train(args.train_start_epoch, args.train_end_epoch, args.batch_size, args.train_ckp_freq)
#     learner.validate(args.val_start_epoch, args.val_end_epoch, args.batch_size, args.val_ckp_freq)

# if __name__ == '__main__':
#     main()