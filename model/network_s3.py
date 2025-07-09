# -*- coding: utf-8 -*-

"""
❗❗❗❗❗❗#此py作用：第三阶段所需要的网络模块
"""
import torch
from torch.nn import init
import torch.nn as nn
import numpy as np
import os
import scipy
import torch.nn.functional as fun
from einops import rearrange
def init_weights(net, init_type, gain):
    print('in init_weights')
    def init_func(m):
        classname = m.__class__.__name__
        #print(classname,m,'_______')
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'mean_space':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1/(height*weight))
            elif init_type == 'mean_channel':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1/(channel))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    
    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net,device, init_type, init_gain,initializer):
    print('in init_net')
    net.to(device)  #gpu_ids[0] 是 gpu_ids列表里面的第一个int值
    if initializer :
        #print(2,initializer)
        init_weights(net,init_type, init_gain)
    else:
        print('Spectral_downsample with default initialize')
    return net

# 定义 CovBlock 和 BandSelectBlock 类
class CovBlock(nn.Module):
    def __init__(self, feature_dimension, features_num, hidden_dim, dropout=0.05):
        super().__init__()
        self.cov_mlp = nn.Sequential(
            nn.Linear(feature_dimension, feature_dimension),
            nn.Dropout(dropout, inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(feature_dimension, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, features_num),
        )

    def forward(self, x):  # x: [B, S, C], S=H*W
        # 1. 中心化
        x = x - x.mean(dim=-1, keepdim=True)
        # 2. 通道协方差
        cov = x.transpose(-2, -1) @ x  # [B, C, C] :与其本身相乘，结果为对角阵(通道协方差矩阵)
        # 2.1 协方差归一化
        cov_norm = torch.norm(x, p=2, dim=-2, keepdim=True)         # [B, 1, C]: 取范数
        cov_norm = cov_norm.transpose(-2, -1) @ cov_norm       # [B, C, C] : 取范数的平方
        cov = cov / (cov_norm + 1e-6)   # [B, C, C] : 归一化
        # 3. 取主对角线作为输入特征
        cov_diag = torch.diagonal(cov, dim1=-2, dim2=-1)  # [B, C]
        if cov_diag.dim() == 1:   # 如果batch为1
            cov_diag = cov_diag.unsqueeze(0)    # [1, C]
        # 4. 通过MLP得到每个通道权重
        return self.cov_mlp(cov_diag)   # [B, features_num]



class BandSelectBlock(nn.Module):
    def __init__(self, feature_dimension, features_num):
        super().__init__()
        self.CovBlockList = nn.ModuleList([
            CovBlock(feature_dimension, feature_dimension, round(feature_dimension * 0.6), 0)
            for _ in range(features_num)
        ])
        self.global_covblock = CovBlock(features_num, 1, features_num, 0)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, feature_maps):
        """
        feature_maps: List of [B, C, H, W]（每一路必须空间尺寸一致，如已对齐）
        融合输出: [B, C, H, W]
        """
        H = feature_maps[0].shape[2]    # 输入特征的空间尺寸
        W = feature_maps[0].shape[3]    # 输入特征的空间尺寸
        C_weights = []  # 各通道权重

        # 对每个输入特征做CovBlock通道加权
        for feature_map, block in zip(feature_maps, self.CovBlockList):
            x_input = rearrange(feature_map, 'B C H W -> B (H W) C') / (H * W - 1)  # [B, H*W, C]
            weight = block(x_input).squeeze(-1)    # [B, C]
            C_weights.append(weight)    # [B, C]

        weight_matrix = torch.stack(C_weights, dim=1)  # [B, features_num, C]
        feature_maps = torch.stack(feature_maps, dim=1) # [B, features_num, C, H, W]

        # 对每个特征逐通道加权
        output = weight_matrix.unsqueeze(-1).unsqueeze(-1) * feature_maps  # [B, features_num, C, H, W]

        # 计算各特征融合权重（全局空间池化后做CovBlock和softmax）
        global_weight = self.global_pool(feature_maps).squeeze(-1).squeeze(-1)  # [B, features_num, C]
        global_weight = torch.softmax(self.global_covblock(global_weight.transpose(-1, -2)), dim=-2) # [B, features_num, 1]

        # 融合所有输入特征
        output = torch.sum(output * global_weight.unsqueeze(-1).unsqueeze(-1), dim=1)  # [B, C, H, W]
        return output




# 简化版Transformer分支
class SimpleSwinTransformerBlock(nn.Module):
    def __init__(self, in_ch, embed_dim=64, num_layers=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, 1)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=4),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
                nn.Conv2d(embed_dim, embed_dim, 1),
                nn.GELU()
            ) for _ in range(num_layers)
        ])
        self.out_proj = nn.Conv2d(embed_dim, in_ch, 1)

    def forward(self, x):
        x = self.proj(x)    # [B, C, H, W]
        for layer in self.layers:
            x = x + layer(x)    # [B, C, H, W]
        x = self.out_proj(x)    # [B, C, H, W]
        return x

# U-Net分支（可复用你原有的 double_u_net_skip 单分支部分/简化版U-Net）
class SimpleUNet(nn.Module):
    def __init__(self, in_ch, base_ch=64):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch*2, 3, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch*4, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_ch*4, base_ch*2, 3, padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch, 3, padding=1), nn.ReLU())
        self.out = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], 1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], 1)
        d1 = self.dec1(d1)
        out = self.out(d1)
        return out

# 新的非对称双流结构
class DualBranchAsymNet(nn.Module):
    def __init__(self, in_ch, args):
        super().__init__()
        self.trans_branch = SimpleSwinTransformerBlock(in_ch)
        self.unet_branch = SimpleUNet(in_ch)
        self.fusion = BandSelectBlock(in_ch, 2) # 2路融合
        self.out_proj = nn.Conv2d(in_ch, in_ch, 1) # 输出高光谱通道不变

    def forward(self, x1, x2):
        feat1 = self.trans_branch(x1)
        feat2 = self.unet_branch(x2)
        fused = self.fusion([feat1, feat2])
        out = self.out_proj(fused)
        return out, feat1, feat2


if __name__ == "__main__":
    pass

    
