import torch
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
        # only different for dimension and position, + 1 is for the classficiatin head

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        x = self.projection(x)
        cls_token = repeat(self.cls, '() 1 Em -> B 1 Em', B=B)
        # cls_token are the same for the one Batch of imgs, I think it is for efficiency
        x = torch.cat([cls_token, x], dim=1)
        x += self.pos 
        return x


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

        assert self.em_size % self.num_heads == 0, "embedding size should be divided by #heads"
        channels = self.em_size//self.num_heads
 
        self.conv1d_list = nn.ModuleList([])
        for k, s, p in zip(self.kernel_size_group, self.stride_group, self.padding_group):
            self.conv1d_list.append(nn.Conv1d(in_channels = channels, out_channels = channels, 
                                    kernel_size=k, stride=s, padding=p)
                                    )
            # in/out_channels should have the same size of embedding                             
        
    def forward(self, x):
        device = x.device

        x = torch.permute(x, (0 ,2, 1)) # x is of (B, Em, N+1)
        x = rearrange(x, 'b (h d) n-> h b d n', h = self.num_heads)
        
        for h in range(self.num_heads):
            conv1d = self.conv1d_list[h].to(device)
            x[h] = conv1d(x[h])
        x = rearrange(x, 'h b d n->b n (h d)')
        self.projection = self.projection.to(device)
        out = self.projection(x)
    
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
        




class EncoderBlock_Conv1D(nn.Sequential):
    def __init__(self,
                 em_size: int=512, 
                 forward_expansion: int = 4,
                 forward_drop_out: float = 0.,
                 **kwargs # padding = 1 means both begining and endding with 1 padding 
                 ): 
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(em_size),      
                MultiHeadConv1D(em_size, **kwargs),
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
    def __init__(self, depth: int = 4, **kwargs):
        super().__init__(*[EncoderBlock_Conv1D(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, em_size: int = 256, n_classes: int = 10):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(em_size), 
            nn.Linear(em_size, n_classes))


# complete model of modified conv1d model
class CoViT(nn.Sequential):
    def __init__(self,     
                in_channels: int = 3,
                patch_size: tuple = (16, 16),
                em_size: int = 512,
                img_size: tuple = (224, 224),
                depth: int = 4,
                n_classes: int = 10,
                **kwargs):
        super().__init__(
            Embedding(in_channels, img_size, patch_size, em_size),
            Encoder(depth, em_size=em_size, **kwargs),
            ClassificationHead(em_size, n_classes)
        )
