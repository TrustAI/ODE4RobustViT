from numpy import save
import torch.nn.functional as F
import torch
from tqdm import tqdm

class Learner:
    '''
        This is to define a learning process include training from skcreth and finetuning

        Args
        dataloaders: train_loader 
        named_network: (network name, network)
        loss_fn: loss function to train the model 
        optimizer: the optimizer to be used 
        lr_scheduler: learning scheduler   
        device: cpu or gpu 
    '''
    def __init__(self, train_loader, named_network, \
                optimizer, lr_scheduler=None, loss_fn = F.cross_entropy, \
                device = None
                ):

        self.train_loader = train_loader
        self.net_name, self.net = named_network
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.device = device
    
    def epoch_run(self, loader, backward_pass=False):
        '''
        this method can be used only for training but validating
        '''
        total_loss = 0
        total_corrects = 0
        
        num_of_batches = 0
        num_of_datas = 0
        for imgs, labels in tqdm(loader):
            num_of_batches += 1
            num_of_datas += len(imgs)

            imgs, labels = imgs.to(self.device), labels.to(self.device)

            logits = self.net(imgs)
            loss = self.loss_fn(logits, labels)
            if backward_pass:
                is_sam = self.optimizer.__class__.__name__ == 'SAM' # SAM use a different procedure to optimize
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
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
 
            preds = logits.argmax(dim=1)
            total_loss += loss.detach().item() 
            total_corrects += preds.eq(labels).sum().detach().item()

        # average loss and acc
        loss = total_loss/num_of_batches
        acc = total_corrects/num_of_datas

        return loss, acc


    def train(self, epochs = 201, save_frequency = 5, verbose = True):
        
        '''
        args:
        epochs: is of a integer indicating training from start or a tulpe indicating training from (start, end)
        save_frequency: if is None then not saving 
        verbose: showing the training result  
        '''
        if len(epochs) == 2:
            epo_interval = range(epochs[0], epochs[1]) 
        else:
            epo_interval = range(*epochs)

        self.net.train() # training mode
        for epoch in epo_interval:
            train_loss, train_acc = self.epoch_run(self.train_loader, backward_pass=True)
            if verbose:
                print(f'train: epoch {epoch} loss {train_loss:.3f}, acc {train_acc:.3f}')
            if save_frequency is not None:
                if epoch % save_frequency == 0:
                    torch.save(self.net.state_dict(), f'./ckp_zoo/{self.net_name}_epoch{epoch}.pth')
            

    def evaluate(self, dataloader):
                
        self.net.eval() # evaluation model
        with torch.no_grad():
            loss, acc  = self.epoch_run(dataloader)
        return loss, acc

            

