import torch
from tqdm import tqdm

def max_sv_compute(func, x):
    
    _x = x.detach().clone().requires_grad_()

    jacob = torch.autograd.functional.jacobian(func, _x)
    jacob_dim = _x[0].shape[0]*_x[0].shape[1]
    jacob = jacob.reshape([jacob_dim,jacob_dim])

    # calculate the singular value 
    svdvals = torch.linalg.svdvals(jacob)
    
    return torch.topk(svdvals, 1)[0]

def max_sv_estimate(func, x):
    '''
    x is a matrix of size (B,N,D)
    '''
    _x = x.detach().clone().requires_grad_() # make the inputs as leaf of computational graph
    
    y = func(_x)

    row_abs_sum = []
    col_abs_sum = 0
    for i in tqdm(range(_x.shape[1])):
        for j in range(_x.shape[2]):
            grad = torch.autograd.grad(inputs=_x, outputs=y[0,i,j], retain_graph=True)[0].detach().clone()

            row_abs_sum.append(grad.abs().sum()) # infinity norm 
            col_abs_sum += grad.flatten().abs() # L_1 norm 

    row_abs_sum = torch.tensor(row_abs_sum)
    upper_bound = (row_abs_sum.max() * col_abs_sum.max()).sqrt()
    return upper_bound





    

