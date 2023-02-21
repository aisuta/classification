import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import math
from torch.nn.modules.batchnorm import _BatchNorm


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x




class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict

class MiniVitv2(nn.Module):
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
        self.layer1 = nn.ModuleList([Block(96,3),Block(96,3)])
        self.dw1 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU()
        )
        self.layer2 = nn.ModuleList([Block(192,6),Block(192,6)])
        self.dw2 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        self.layer3 = nn.ModuleList([Block(384,12),Block(384,12)])
        self.dw3 = nn.Sequential(
            nn.Conv2d(384, 768, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU()
        )
        self.layer4 = nn.ModuleList([Block(768,24),Block(768,24)])
    
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=768, out_features=5, bias=True)
        self.norm_eval = norm_eval
    def forward(self,x):
        outs = []
        x = self.embed(x)
        
        b,c,h,w = x.shape
        x = x.reshape(b,c,-1).transpose(1,2)
        for layer in self.layer1:
            x = layer(x,h,w)
        x = x.transpose(1,2).reshape(b,c,h,w)
        outs.append(x)
        
        x = self.dw1(x)
        
        b,c,h,w = x.shape
        x = x.reshape(b,c,-1).transpose(1,2)
        for layer in self.layer2:
            x = layer(x,h,w)
        x = x.transpose(1,2).reshape(b,c,h,w)
        outs.append(x)
        
        x = self.dw2(x)
        
        b,c,h,w = x.shape
        x = x.reshape(b,c,-1).transpose(1,2)
        for layer in self.layer3:
            x = layer(x,h,w)
        x = x.transpose(1,2).reshape(b,c,h,w)
        outs.append(x)
        
        x = self.dw3(x)
        
        b,c,h,w = x.shape
        x = x.reshape(b,c,-1).transpose(1,2)
        for layer in self.layer4:
            x = layer(x,h,w)
        x = x.transpose(1,2).reshape(b,c,h,w)
        outs.append(x)
   
        return tuple(outs)
    def forward_features(self,x,need_fea=False):
        if need_fea:
            outs = []
            x = self.embed(x)
            
            b,c,h,w = x.shape
            x = x.reshape(b,c,-1).transpose(1,2)
            for layer in self.layer1:
                x = layer(x,h,w)
            x = x.transpose(1,2).reshape(b,c,h,w)
            outs.append(x)
            
            x = self.dw1(x)
            
            b,c,h,w = x.shape
            x = x.reshape(b,c,-1).transpose(1,2)
            for layer in self.layer2:
                x = layer(x,h,w)
            x = x.transpose(1,2).reshape(b,c,h,w)
            outs.append(x)
            
            x = self.dw2(x)
            
            b,c,h,w = x.shape
            x = x.reshape(b,c,-1).transpose(1,2)
            for layer in self.layer3:
                x = layer(x,h,w)
            x = x.transpose(1,2).reshape(b,c,h,w)
            outs.append(x)
            
            x = self.dw3(x)
            
            b,c,h,w = x.shape
            x = x.reshape(b,c,-1).transpose(1,2)
            for layer in self.layer4:
                x = layer(x,h,w)
            x = x.transpose(1,2).reshape(b,c,h,w)
            outs.append(x)

            return outs, outs[-1].mean([2, 3])
        else:
            x = self.embed(x)
            
            b,c,h,w = x.shape
            x = x.reshape(b,c,-1).transpose(1,2)
            for layer in self.layer1:
                x = layer(x,h,w)
            x = x.transpose(1,2).reshape(b,c,h,w)
            
            x = self.dw1(x)
            
            b,c,h,w = x.shape
            x = x.reshape(b,c,-1).transpose(1,2)
            for layer in self.layer2:
                x = layer(x,h,w)
            x = x.transpose(1,2).reshape(b,c,h,w)
            
            x = self.dw2(x)
            
            b,c,h,w = x.shape
            x = x.reshape(b,c,-1).transpose(1,2)
            for layer in self.layer3:
                x = layer(x,h,w)
            x = x.transpose(1,2).reshape(b,c,h,w)
            
            x = self.dw3(x)
            
            b,c,h,w = x.shape
            x = x.reshape(b,c,-1).transpose(1,2)
            for layer in self.layer4:
                x = layer(x,h,w)
            x = x.transpose(1,2).reshape(b,c,h,w)

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
    
def minivitv2(pretrained: bool, *args, **kwargs):
    model = MiniVitv2(*args, **kwargs)

    
    return model