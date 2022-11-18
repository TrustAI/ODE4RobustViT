import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce


class Embedding(nn.Module):
    '''
    Embedding: 
        Input: img(B, C, H, W)
        Output: img(B, N+1, Em), Conv2D(in_channel=C, out_channel=Em_size, kernel_size=patch_size, stride=patch_size)
    
    Arguments: in_channel=3, img_size=[224,224], patch_size=[16,16], Em_size=512 
    '''
    def __init__(self, in_channels: int=3, img_size: tuple=(224, 224), patch_size: tuple=(16, 16), em_size: int=256):
        super().__init__()
        self.patch_size=patch_size
        self.projection=nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=em_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('B Em H W->B (H W) Em')
            )
        self.cls = nn.Parameter(torch.randn([1, 1, em_size]))
        # class token are only different in em_size
        self.pos = nn.Parameter(torch.randn([(img_size[0]//patch_size[0])*(img_size[1]//patch_size[1]) + 1, em_size]))
        # only different for dimension and position, + 1 is for the class token

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        x = self.projection(x)
        cls_token = repeat(self.cls, '() 1 Em -> B 1 Em', B=B)
        # cls_token are the same for the one Batch of imgs, I think it is for efficiency
        x = torch.cat([cls_token, x], dim=1)
        x += self.pos 
        return x


class MultiHeadAttention(nn.Module):
    '''
    Multihead Attention:
        Input: img(B N+1 Em)
        Output: softmax(QK^T/sqrt(d_K))V = img(B N+1 Em)
        
    Arguments: d_k, d_v, #heads, drop_out
    '''
    def __init__(self, em_size: int=512, d_K: int=512, d_V: int=512, num_heads: int=4, drop_out: float=0):
        super().__init__()
        self.num_heads = num_heads
        self.d_K = d_K
        self.qk = nn.Linear(em_size, d_K*2)
        self.v = nn.Linear(em_size, d_V)
        self.att_drop = nn.Dropout(drop_out)
        self.projection = nn.Linear(d_V, em_size)


    def forward(self, x: Tensor) -> Tensor:
        qk = rearrange(self.qk(x), 'b n (h d qk)->qk b h n d', h=self.num_heads, qk=2)
        Q, K= qk[0], qk[1]
        V = rearrange(self.v(x), 'b n (h d)->b h n d', h=self.num_heads)
        QK_T = torch.einsum('bhqd, bhkd->bhqk', Q, K)
        # matrics/tensor dot product via einsum

        d_k = self.d_K//self.num_heads

        att = F.softmax(QK_T/(d_k)**0.5, dim=-1)
        att = self.att_drop(att)

        out = torch.einsum('bhqk, bhkv->bhqv', att, V)
        out = rearrange(out, 'b h n d-> b n (h d)')
        out = self.projection(out)
        return out


# residual adding block 
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        '''
        fn is actuall nn.Sequential which can be regarded as a function
        '''
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

# feed forward block
class FeedForwardBlock(nn.Sequential):
    def __init__(self, em_size: int=256, expansion: int = 4, drop_out: float = 0.):
        super().__init__(
            nn.Linear(em_size, expansion * em_size),
            nn.GELU(), 
            nn.Dropout(drop_out),
            nn.Linear(expansion * em_size, em_size),
        )
        

# encoder block 
class EncoderBlock(nn.Sequential):
    def __init__(self,
                 em_size: int=512, 
                 forward_expansion: int = 4,
                 forward_drop_out: float = 0.,
                 **kwargs): # kwarg cannot be used twice, hence arguments of MLP still have to be initialized
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(em_size), #em_size is the number and dimension for gamma and beta,
                                       # which is the region of calculation of mean and variacne     
                MultiHeadAttention(em_size, **kwargs),
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(em_size),
                FeedForwardBlock(
                    em_size, expansion=forward_expansion, drop_out=forward_drop_out 
                ),
            ))
        )



# a bounch of encoders 
class Encoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[EncoderBlock(**kwargs) for _ in range(depth)])

# classification head 
class ClassificationHead(nn.Sequential):
    def __init__(self, em_size: int = 256, n_classes: int = 10):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(em_size), 
            nn.Linear(em_size, n_classes))

# complete model of vit
class ViT(nn.Sequential):
    def __init__(self,     
                in_channels: int = 3,
                patch_size: tuple = (16, 16),
                em_size: int = 512,
                img_size: tuple = (224, 224),
                depth: int = 12,
                n_classes: int = 10,
                **kwargs):
        super().__init__(
            Embedding(in_channels, img_size, patch_size, em_size),
            Encoder(depth, em_size=em_size, **kwargs),
            ClassificationHead(em_size, n_classes)
        )
    