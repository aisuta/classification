from torch.nn.modules.batchnorm import _BatchNorm
import torch             # torch基础库
import torch.nn as nn    # torch神经网络库
import torch.nn.functional as F

class CAFormer(nn.Module):
    def __init__(self,dim,n_head):
        super().__init__()
        self.qk = nn.Conv2d(dim, dim*2, 3, 1, 1, groups=dim)
        self.v = nn.Conv2d(dim, dim, 1, 1)
        self.att = nn.Softmax(dim=-1)
        self.proj = nn.Conv2d(dim,dim,1,1)
        self.nb = nn.BatchNorm2d(dim)
        self.act = nn.SiLU()
        self.n_heads = n_head
        
    def divide_window(self,x,ks):
        b,c,h,w = x.shape
        x = F.unfold(x,ks,1,ks//2,1)
        return x
    
    def forward(self,x):
        b,c,h,w = x.shape
        # b c h*w 9 unfold
        # b h*w c * 9 @ b h*w 9 c -> b h*w c c -> softmax 
        # b h*w c c @ b h*w c 1 -> b h*w c -> b c h w
        ide = x
        qk = self.qk(x).reshape(b,2,c,h,w).transpose(0,1)
        q = qk[0]
        k = qk[1]
        B = self.n_heads *b
        C = c//self.n_heads
        q = q.reshape(B,C,h,w)
        k = k.reshape(B,C,h,w)
        
        q = self.divide_window(q,3).reshape(B,C,h*w,-1).transpose(1,2)
        k = self.divide_window(k,3).reshape(B,C,h*w,-1).transpose(1,2).transpose(2,3)
        v = self.v(x).reshape(B,C,h,w).transpose(1,2).transpose(2,3).reshape(B,-1,C,1)
        qk = q @ k
        qk = self.att(qk)
        qkv = qk @ v
        qkv = qkv.transpose(1,2).reshape(b,c,h,w)
        x_update = self.proj(qkv)
        
        x = self.nb(x_update)
        x = self.act(x)
        
        return x+ ide
        
        
class StageCFormer(nn.Module):
    def __init__(self,incs,oucs,n_head):
        super().__init__()
        self.ca = CAFormer(incs,n_head)
        self.ln1 = None
        self.ln2 = None
        self.ln3 = None
        self.ln4 = None
        self.h = None
        self.w = None
        self.ffn1 = nn.Conv2d(incs,4*oucs,1,1)
        self.ffn2 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(4*oucs,oucs,1,1)
        )
        self.act = nn.SiLU()
        
    def forward(self,x):
        ide = x
        x = self.ca(x)
        b,c,h,w = x.shape
        if h != self.h or w != self.w:
            self.w = w
            self.h = h
            self.ln1 = nn.LayerNorm((h,w)).to(x.device)
            self.ln2 = nn.LayerNorm((h,w)).to(x.device)
            self.ln3 = nn.LayerNorm((h,w)).to(x.device)
            self.ln4 = nn.LayerNorm((h,w)).to(x.device)
        x = self.ln1(x)
        x = x+ide 
        ide = x
        x = self.ffn1(x)
        x = self.ffn2(x)
        x = self.act(x)
        x = x+ide 
        return x
    
    
    
class MiniCAF(nn.Module):
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
        self.layer1 = nn.Sequential(
            StageCFormer(96,96,3),StageCFormer(96,96,3)
        )
        self.dw1 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            StageCFormer(192,192,6),StageCFormer(192,192,6)
        )
        self.dw2 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            StageCFormer(384,384,12),StageCFormer(384,384,12)
        )
        self.dw3 = nn.Sequential(
            nn.Conv2d(384, 768, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            StageCFormer(768,768,24),StageCFormer(768,768,24)
        )
    
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
    
def minicaf(pretrained: bool,*args, **kwargs):
    model = MiniCAF(*args, **kwargs)

    
    return model