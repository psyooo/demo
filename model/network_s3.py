# -*- coding: utf-8 -*-

"""
❗❗❗❗❗❗#此py作用：所需要的网络模块
"""
import torch
from torch.nn import init
import torch.nn as nn
import numpy as np
import os
import scipy
import torch.nn.functional as F
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




# 定义深度可分离卷积模块
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# 定义 AttentionGate 类
class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels):
        super().__init__()
        self.theta = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.phi = nn.Conv2d(gating_channels, in_channels // 2, kernel_size=1)
        self.psi = nn.Conv2d(in_channels // 2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        theta_x = self.theta(x)
        phi_g = self.phi(g)
        # 调整 phi_g 的形状，使其与 theta_x 一致
        phi_g = F.interpolate(phi_g, size=theta_x.shape[2:], mode='bilinear', align_corners=True)
        f = F.relu(theta_x + phi_g)
        psi_f = self.psi(f)
        attention = self.sigmoid(psi_f)
        return attention * x

# ============================== 空洞卷积模块定义 =================================
class DilatedConvBlock(nn.Module):
    def __init__(self, in_ch, num_layers=3, dilation_rates=[1, 2, 4]):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = dilation_rates[i % len(dilation_rates)]  # 循环使用空洞率
            layers.extend([
                nn.Conv2d(in_ch, in_ch, 3, padding=dilation, dilation=dilation),  # 空洞卷积
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True)
            ])
        self.layers = nn.Sequential(*layers)
        self.out_proj = nn.Conv2d(in_ch, in_ch, 1)

    def forward(self, x):
        residual = x
        x = self.layers(x)
        x = self.out_proj(x)
        return x + residual
# ============================== 空洞卷积模块定义结束 =================================



# U-Net分支（可复用你原有的 double_u_net_skip 单分支部分/简化版U-Net）
class SimpleUNet(nn.Module):
    def __init__(self, in_ch, base_ch=64):
        super().__init__()
        self.enc1 = nn.Sequential(
            DepthwiseSeparableConv(in_ch, base_ch, 3, padding=1), nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            DepthwiseSeparableConv(base_ch, base_ch * 2, 3, padding=1), nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            DepthwiseSeparableConv(base_ch * 2, base_ch * 4, 3, padding=1), nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.attn_gate2 = AttentionGate(base_ch * 2, base_ch * 4)
        self.attn_gate1 = AttentionGate(base_ch, base_ch * 2)
        self.dec2 = nn.Sequential(
            DepthwiseSeparableConv(base_ch * 4, base_ch * 2, 3, padding=1), nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            DepthwiseSeparableConv(base_ch * 2, base_ch, 3, padding=1), nn.ReLU()
        )
        self.relu = nn.ReLU()
        self.LReLU = nn.LeakyReLU(0.2)
        self.out = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x):
        e1 = self.relu(self.enc1(x))

        e2 = self.relu(self.enc2(self.pool(e1)))
        e3 = self.relu(self.enc3(self.pool(e2)))
        d2 = self.up2(e3)
        e2_attn = self.attn_gate2(e2, e3)
        d2 = torch.cat([d2, e2_attn], 1)
        d2 = self.relu(self.dec2(d2))
        d1 = self.up1(d2)
        e1_attn = self.attn_gate1(e1, d2)
        d1 = torch.cat([d1, e1_attn], 1)
        d1 = self.relu(self.dec1(d1))
        out = self.out(d1)
        return out

# 新的非对称双流结构
class DualBranchAsymNet(nn.Module):
    def __init__(self, in_ch, args):
        super().__init__()
        self.trans_branch = DilatedConvBlock(in_ch)
        self.unet_branch = SimpleUNet(in_ch)
        self.fusion = BandSelectBlock(in_ch, 2) # 2路融合
        self.out_proj = nn.Conv2d(in_ch, in_ch, 1) # 输出高光谱通道不变

    def forward(self, x1, x2):
        feat1 = self.trans_branch(x1.clone())
        feat2 = self.unet_branch(x2.clone())
        # fused = self.fusion([feat1, feat2])
        # out = self.out_proj(fused)
        return feat1, feat2


if __name__ == "__main__":
    pass

    
