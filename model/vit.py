import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn.modules.batchnorm import _BatchNorm
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), 
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
class Attention(nn.Module):              
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)           # (b, n(65), dim*3) ---> 3 * (b, n, dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)          # q, k, v   (b, h, n, dim_head(64))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
class VITransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    
    def forward(self, x):
        b,c,h,w = x.shape
        x = x.permute(0,2,3,1).reshape(b,-1,c)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = x.transpose(1,2).reshape(b,c,h,w)
        return x
# from utils import isnan
class MiniVit(nn.Module):
    def __init__(self,norm_eval=False):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        self.layer1 = VITransformer(96,2,3,32,96*4)
        self.dw1 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU()
        )
        self.layer2 = VITransformer(192,2,6,32,192*4)
        self.dw2 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        self.layer3 = VITransformer(384,2,12,32,384*4)
        self.dw3 = nn.Sequential(
            nn.Conv2d(384, 768, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU()
        )
        self.layer4 = VITransformer(768,2,24,32,768*4)
    
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=768, out_features=5, bias=True)
        self.norm_eval = norm_eval
    def forward_features(self,x,need_fea=False):
        if need_fea:
            outs = []
            x = self.embed(x)
            x = self.layer1(x)
            outs.append(x)
            x = self.dw1(x)
            x = self.layer2(x)
            outs.append(x)
            x = self.dw2(x)
            x = self.layer3(x)
            outs.append(x)
            x = self.dw3(x)
            x = self.layer4(x)
            outs.append(x)
            return outs, outs[-1].mean([2, 3])
        else:
            x = self.embed(x)
            x = self.layer1(x)
           
            x = self.dw1(x)
            x = self.layer2(x)
           
            x = self.dw2(x)
            x = self.layer3(x)
            
            x = self.dw3(x)
            x = self.layer4(x)
            x = x.mean([2, 3])
        return x
    
    def _forward_impl(self, x, need_fea=False):
        if need_fea:
            features, features_fc = self.forward_features(x, need_fea)
            return features, features_fc, self.fc(features_fc)
        else:
            x = self.forward_features(x)
            x = self.fc(x)
            return x
    def forward(self, x, need_fea=False):
        return self._forward_impl(x, need_fea)
    def switch_to_deploy(self):
        pass
    def cam_layer(self):
        return self.layer4
    
def minivit(pretrained: bool,*args, **kwargs):
    model = MiniVit(*args, **kwargs)

    
    return model