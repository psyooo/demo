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
        # print(classname,m,'_______')
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
                m.weight.data.fill_(1 / (height * weight))
            elif init_type == 'mean_channel':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1 / (channel))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, device, init_type, init_gain, initializer):
    print('in init_net')
    net.to(device)  # gpu_ids[0] 是 gpu_ids列表里面的第一个int值
    if initializer:
        # print(2,initializer)
        init_weights(net, init_type, init_gain)
    else:
        print('Spatial_upsample with default initialize')
    return net



def create_inverse_PSF(sigma, shape=(3, 3), device='cpu'):
    """
    直接在网络文件里生成逆PSF核。
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = torch.meshgrid(torch.arange(-m, m + 1), torch.arange(-n, n + 1), indexing='ij')
    h = torch.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    # 可以简单地认为高斯核就是近似的逆PSF（物理上更建议用Wiener逆，但常用这种简化）
    return h.to(device)

class InversePSF(nn.Module):
    def __init__(self, inv_psf, scale):
        """
        inv_psf: torch.Tensor, shape (k, k)
        scale: int, 卷积核大小和stride（等于目标上采样倍数）
        """
        super().__init__()
        k = inv_psf.shape[0]
        self.convT = nn.ConvTranspose2d(
            1, 1, kernel_size=k, stride=scale, padding=0, bias=False
        )
        self.convT.weight.data[0, 0] = inv_psf.float()
        self.convT.weight.requires_grad = True

    def forward(self, x):
        # x: [B, 1, h, w]
        return self.convT(x)

class MatrixDotLR2HR(nn.Module):
    def __init__(self, inv_psf, scale, max_channels):
        super().__init__()
        self.n_channels = max_channels
        self.band_layers = nn.ModuleList([
            InversePSF(inv_psf, scale) for _ in range(max_channels)
        ])

    def forward(self, x):
        # x: [B, C, h, w]
        batch, channel, height, width = x.size()
        outs = [self.band_layers[i](x[:, i:i+1, :, :]) for i in range(channel)]  # [B, 1, H, W]
        out = torch.cat(outs, dim=1)  # [B, C, H, W]
        return out

class SpaceUpNet(nn.Module):
    """双流空间超分主干，与EDIP双流结构一致。支持两路输入，分支可以同构或异构"""
    def __init__(self, in_ch, out_ch, scale=2):
        super().__init__()
        # 分支1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False),
            nn.Conv2d(64, out_ch, 3, padding=1)
        )
        # 分支2（可以同构，也可以设计成不同结构）
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False),
            nn.Conv2d(64, out_ch, 3, padding=1)
        )
        # 可以插入交互/融合/残差等模块

    def forward(self, x1, x2):
        # x1, x2 分别为两路输入（如Lr-MSI1和Lr-MSI2）
        y1 = self.branch1(x1)
        y2 = self.branch2(x2)
        return y1, y2
