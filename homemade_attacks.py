import torch 
import torch.nn as nn
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
# img_transform
def img_transform(imgs): # imgs is of size (B, C, H, W)
  img = []
  for i in range(3):
      img.append(imgs[:,i,:,:])

  img = torch.stack(img, dim=3)
  return img

# plot the images 
def imgs_save(adv_imgs, labels, preds, layout=(2,4), path=None):
    # adv_imgs is of (B C W H)
    label_dict = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck' ]
    adv_imgs_trans = img_transform(adv_imgs).to('cpu')
    preds = preds.argmax(dim=1)

    plt.figure(figsize=(10,5))
    for i in range(layout[0]*layout[1]):
        plt.subplot(layout[0], layout[1], i+1)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title("{} -> {}".format(label_dict[labels[i]], label_dict[preds[i]]))
        plt.imshow(adv_imgs_trans[i])
    plt.tight_layout()
    plt.savefig(path)



class Attack:

    def __init__(self, named_network, dataloader, target, Lp):
        self.net_name, self.net = named_network
        self.dataloader = dataloader
        self.target = target # number that indicate the target or 'all' for untarget attack
        self.Lp = Lp

    def attack(self, method, **kwargs):
        # run the attack for the whole attaking img set
        succ = 0
        self.train()
        for imgs, labels in self.dataloader:
            
            logits = self.net(imgs)
            preds = logits.argmax(dim=1)
            idx = preds == labels

            if method == 'fgsm':
                perturbed_X = self._fgsm(imgs[idx], labels[idx], self.net,**kwargs)
            if method == 'pgd':
                pass
            if method == 'cw':
                pass

            succ += self.net(perturbed_X).argmax(dim=1).eq(labels).sum().detach().item()

        return succ


    @staticmethod
    def _fgsm(imgs, labels, model, epsilon):
        '''
        This the function for attacking a batch of images, given ground truth ys
        Arg: 
            imgs are the input image of size(B,C,H,W)
            labels are the ground truth labels
            model is the network to be attacked 
            epsilon is the L_/infty radius for fgsm attacking
        Output
            the perturbed images
        ''' 
        delta = torch.zeros_like(imgs, requires_grad=True)
        loss = nn.CrossEntropyLoss()(model(imgs + delta), labels)
        loss.backward()
        perturbation = epsilon * delta.grad.detach().sign()
        perturbed_imgs = torch.clamp(imgs + perturbation, 0, 1)
        return perturbed_imgs
    
    @staticmethod
    def _pgd(imgs, labels, model, target_label, Lp, epsilon, alpha, iter_max, restarts_num = 1, zero_init = False):
        
        max_loss = torch.zeros(labels.shape[0]).to(labels.device)
        max_delta = torch.zeros_like(imgs)

        for i in range(restarts_num):
            if zero_init:
                delta = torch.zeros_like(imgs, requires_grad=True) 
            else:
                delta = torch.rand_like(imgs, requires_grad=True) # uniform distribution
                delta.data = delta.data * 2 * epsilon - epsilon #[-epsilon, epsilon]

            for j in range(iter_max):
                # different setup of loss function
                if target_label == 'all':
                    loss = -nn.CrossEntropyLoss()(model(imgs + delta), labels)
                else:
                    pred_labels = model(imgs + delta)
                    loss = 2*pred_labels[:,target_label].sum() - pred_labels.sum()
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
                    delta.data = torch.min(torch.max(delta.detach(), -imgs), 1-imgs)
                    # clip X+delta to [0,1]
                    delta.data *= epsilon / \
                    ((delta.detach().abs()**Lp).flatten(start_dim=1).sum(dim=1)[:,None, None, None]**(1/Lp)).clamp(min=epsilon)
                    # projection to the L_p norm ball 

                delta.grad.zero_()

            all_loss = nn.CrossEntropyLoss(reduction='none')(model(imgs+delta),labels)
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
            # each delta in max_delta has the largest loss

            max_loss = torch.max(max_loss, all_loss)
            # torch.max will subtitute to smaller value for each dimension 
        return max_delta

    class _cw:
        pass

    def _save_adv_imgs(self):
        pass 
