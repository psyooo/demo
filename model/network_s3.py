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
        # print("weight_matrix", weight_matrix.shape)
        # print("feature_maps", feature_maps.shape)
        # 对每个特征逐通道加权
        output = weight_matrix.unsqueeze(-1).unsqueeze(-1) * feature_maps  # [B, features_num, C, H, W]

        # 计算各特征融合权重（全局空间池化后做CovBlock和softmax）
        global_weight = self.global_pool(feature_maps).squeeze(-1).squeeze(-1)  # [B, features_num, C]
        global_weight = torch.softmax(self.global_covblock(global_weight.transpose(-1, -2)), dim=-2) # [B, features_num, 1]

        # 融合所有输入特征
        output = torch.sum(output * global_weight.unsqueeze(-1).unsqueeze(-1), dim=1)  # [B, C, H, W]
        return output


########################## double_U_net_skip ############################

def double_U_net_skip(Out_fhsi,Out_fmsi,args, init_type='kaiming', init_gain=0.02,initializer=True ):
    
    net = double_u_net_skip(Out_fhsi,Out_fmsi,args)

    return init_net(net,args.device, init_type, init_gain,initializer)

class double_u_net_skip(nn.Module):
    def __init__(self,Out_fhsi,Out_fmsi,args): 

        super().__init__()


        self.band=args.band
        hidden_dim = self.band // 2
        #self.fusion=fusion
        self.Out_fhsi=Out_fhsi
        self.Out_fmsi=Out_fmsi
        self.scale=[
                              (  self.Out_fhsi.shape[2],self.Out_fhsi.shape[3]  ),
                              (  int(self.Out_fhsi.shape[2]/2),int(self.Out_fhsi.shape[3]/2)  ),
                              (  int(self.Out_fhsi.shape[2]/4), int(self.Out_fhsi.shape[3]/4) )
                              ]
        print(self.scale)
       
        
        
        '''for out_fhsi'''
        self.ex1=nn.Sequential(
        nn.Conv2d(self.Out_fhsi.shape[1],self.band,kernel_size=(5,5),stride=1,padding=(2,2)) ,
        nn.BatchNorm2d(self.band),
        nn.LeakyReLU(0.2, inplace=True) #nn.LeakyReLU(0.2, inplace=True) nn.ReLU(inplace=True) 
                                )
        
        self.ex2=nn.Sequential(
        nn.Conv2d(self.band,self.band,kernel_size=(5,5),stride=1,padding=(2,2)) ,
        nn.BatchNorm2d(self.band),
        nn.LeakyReLU(0.2, inplace=True) #nn.LeakyReLU(0.2, inplace=True)
                                )

        self.ex3=nn.Sequential(
        nn.Conv2d(self.band,self.band,kernel_size=(5,5),stride=1,padding=(2,2)) ,
        nn.BatchNorm2d(self.band),
        nn.LeakyReLU(0.2, inplace=True) #nn.LeakyReLU(0.2, inplace=True)
                                )
        
        self.ex4=nn.Sequential(
        nn.Conv2d(self.band+2,self.band,kernel_size=(5,5),stride=1,padding=(2,2)) ,
        nn.BatchNorm2d(self.band),
        nn.LeakyReLU(0.2, inplace=True) #nn.LeakyReLU(0.2, inplace=True)
                                )
        
        self.ex5=nn.Sequential(
        nn.Conv2d(self.band+2,self.band,kernel_size=(5,5),stride=1,padding=(2,2)) ,
        #nn.Sigmoid()  #nn.LeakyReLU(0.2, inplace=True)
        nn.BatchNorm2d(self.band),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(self.band,self.Out_fhsi.shape[1],kernel_size=(1,1),stride=1,padding=(0,0)) ,
        nn.Sigmoid()
                                )
        
        self.skip1=nn.Sequential(
        nn.Conv2d(self.band,2,kernel_size=(1,1),stride=1,padding=(0,0)) ,
        nn.BatchNorm2d(2),
        nn.LeakyReLU(0.2, inplace=True)
                                )
        
        self.skip2=nn.Sequential(
        nn.Conv2d(self.band,2,kernel_size=(1,1),stride=1,padding=(0,0)) ,
        nn.BatchNorm2d(2),
        nn.LeakyReLU(0.2, inplace=True)
                                )
        '''for out_fhsi'''
        
        '''for out_fmsi'''
        
        self.ex6=nn.Sequential(
        nn.Conv2d(self.Out_fmsi.shape[1],self.band,kernel_size=(5,5),stride=1,padding=(2,2)) ,
        nn.BatchNorm2d(self.band),
        nn.LeakyReLU(0.2, inplace=True) #nn.LeakyReLU(0.2, inplace=True) nn.ReLU(inplace=True) 
                                )
        
        self.ex7=nn.Sequential(
        nn.Conv2d(self.band,self.band,kernel_size=(5,5),stride=1,padding=(2,2)) ,
        nn.BatchNorm2d(self.band),
        nn.LeakyReLU(0.2, inplace=True) #nn.LeakyReLU(0.2, inplace=True)
                                )

        self.ex8=nn.Sequential(
        nn.Conv2d(self.band,self.band,kernel_size=(5,5),stride=1,padding=(2,2)) ,
        nn.BatchNorm2d(self.band),
        nn.LeakyReLU(0.2, inplace=True) #nn.LeakyReLU(0.2, inplace=True)
                                )
        
        self.ex9=nn.Sequential(
        nn.Conv2d(self.band+2,self.band,kernel_size=(5,5),stride=1,padding=(2,2)) ,
        nn.BatchNorm2d(self.band),
        nn.LeakyReLU(0.2, inplace=True) #nn.LeakyReLU(0.2, inplace=True)
                                )
        
        self.ex10=nn.Sequential(
        nn.Conv2d(self.band+2,self.band,kernel_size=(5,5),stride=1,padding=(2,2)) ,
        #nn.Sigmoid()  #nn.LeakyReLU(0.2, inplace=True)
        nn.BatchNorm2d(self.band),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(self.band,self.Out_fmsi.shape[1],kernel_size=(1,1),stride=1,padding=(0,0)) ,
        nn.Sigmoid()
                                )
        
        self.skip3=nn.Sequential(
        nn.Conv2d(self.band,2,kernel_size=(1,1),stride=1,padding=(0,0)) ,
        nn.BatchNorm2d(2),
        nn.LeakyReLU(0.2, inplace=True)
                                )
        
        self.skip4=nn.Sequential(
        nn.Conv2d(self.band,2,kernel_size=(1,1),stride=1,padding=(0,0)) ,
        nn.BatchNorm2d(2),
        nn.LeakyReLU(0.2, inplace=True)
                                )
        '''for out_fmsi'''

        '''CovBlock和BandSelectBlock集成'''
        # 多尺度CovBlock（x1、x3、x5三个点）
        self.covblock1 = CovBlock(self.band, self.band, hidden_dim=self.band // 2)
        self.covblock3 = CovBlock(self.band, self.band, hidden_dim=self.band // 2)
        self.covblock5 = CovBlock(self.band, self.band, hidden_dim=self.band // 2)
        # 多尺度融合
        self.bandselect_fhsi = BandSelectBlock(self.band, 3)  # 3表示要融合x1,x3,x5
        self.fused_proj = None  # 占位
        self.out_fhsi_conv = None  # 如你后面还需要额外1x1卷积，也用None
        # 下分支
        self.covblock6 = CovBlock(self.band, self.band, hidden_dim=self.band // 2)
        self.covblock7 = CovBlock(self.band, self.band, hidden_dim=self.band // 2)
        self.covblock8 = CovBlock(self.band, self.band, hidden_dim=self.band // 2)
        self.bandselect_fmsi = BandSelectBlock(self.band, 3)
        self.fused_proj_fmsi = None
        self.out_fmsi_conv = None
        # self.fused_proj = nn.Conv2d(self.band, self.out_ch, 1)  # 其中self.band=fused通道数，self.out_ch=输出通道数

        # 输出1x1卷积
        # self.out_fhsi_conv = nn.Conv2d(self.band, self.Out_fhsi.shape[1], 1)

    def forward(self, Out_fhsi, Out_fmsi):
        # print("=== [上分支/Out_fhsi] ===")
        # 编码
        x1 = self.ex1(Out_fhsi)  # [B, C, H, W]
        x2 = nn.AdaptiveAvgPool2d(self.scale[1])(x1)
        x3 = self.ex2(x2)
        x4 = nn.AdaptiveAvgPool2d(self.scale[2])(x3)
        x5 = self.ex3(x4)

        # CovBlock加权
        x1_w = self.covblock1(rearrange(x1, 'B C H W -> B (H W) C'))
        x1_w = x1_w.view(x1.shape[0], x1.shape[1], 1, 1)
        x1_weighted = x1 * x1_w + x1
        # print("x1_weighted", x1_weighted.shape)

        x3_w = self.covblock3(rearrange(x3, 'B C H W -> B (H W) C'))
        x3_w = x3_w.view(x3.shape[0], x3.shape[1], 1, 1)
        x3_weighted = x3 * x3_w + x3
        # print("x3_weighted", x3_weighted.shape)

        x5_w = self.covblock5(rearrange(x5, 'B C H W -> B (H W) C'))
        x5_w = x5_w.view(x5.shape[0], x5.shape[1], 1, 1)
        x5_weighted = x5 * x5_w + x5
        # print("x5_weighted", x5_weighted.shape)

        # 解码主路
        up = nn.Upsample(self.scale[1], mode='bilinear')
        s1 = self.skip1(x3_weighted)
        x6 = up(x5_weighted)
        x7 = self.ex4(torch.cat([s1, x6], dim=1))
        up = nn.Upsample(self.scale[0], mode='bilinear')
        s2 = self.skip2(x1_weighted)
        x8 = up(x7)
        out_fhsi = self.ex5(torch.cat([s2, x8], dim=1))
        # print("out_fhsi (decode主路输出)", out_fhsi.shape)

        # 多尺度融合（空间对齐）
        x3_weighted_up = fun.interpolate(x3_weighted, size=self.scale[0], mode='bilinear', align_corners=False)
        x5_weighted_up = fun.interpolate(x5_weighted, size=self.scale[0], mode='bilinear', align_corners=False)

        fused = self.bandselect_fhsi([x1_weighted, x3_weighted_up, x5_weighted_up])
        # print("fused (BandSelectBlock融合)", fused.shape)

        # --- 自动初始化 fused_proj ---
        if self.fused_proj is None:
            in_ch = fused.shape[1]
            out_ch = out_fhsi.shape[1]
            self.fused_proj = nn.Conv2d(in_ch, out_ch, 1).to(fused.device)
            # print(f"自动初始化fused_proj: Conv2d({in_ch}, {out_ch}, 1)")
        fused_proj = self.fused_proj(fused)  # [B, out_ch, H, W]

        # --- 你也可以自动初始化 out_fhsi_conv ---
        if self.out_fhsi_conv is None:
            out_ch = out_fhsi.shape[1]
            self.out_fhsi_conv = nn.Conv2d(out_ch, out_ch, 1).to(out_fhsi.device)
            # print(f"自动初始化out_fhsi_conv: Conv2d({out_ch}, {out_ch}, 1)")

        # 融合最终输出
        Out_fhsi = self.out_fhsi_conv(fused_proj + out_fhsi)  # [B, out_ch, H, W]


        # 下分支编码
        x9 = self.ex6(Out_fmsi)
        x10 = nn.AdaptiveAvgPool2d(self.scale[1])(x9)
        x11 = self.ex7(x10)
        x12 = nn.AdaptiveAvgPool2d(self.scale[2])(x11)
        x13 = self.ex8(x12)

        # CovBlock加权
        x9_w = self.covblock6(rearrange(x9, 'B C H W -> B (H W) C'))
        x9_w = x9_w.view(x9.shape[0], x9.shape[1], 1, 1)
        x9_weighted = x9 * x9_w + x9
        # print("x9_weighted", x9_weighted.shape)

        x11_w = self.covblock7(rearrange(x11, 'B C H W -> B (H W) C'))
        x11_w = x11_w.view(x11.shape[0], x11.shape[1], 1, 1)
        x11_weighted = x11 * x11_w + x11
        # print("x11_weighted", x11_weighted.shape)

        x13_w = self.covblock8(rearrange(x13, 'B C H W -> B (H W) C'))
        x13_w = x13_w.view(x13.shape[0], x13.shape[1], 1, 1)
        x13_weighted = x13 * x13_w + x13
        # print("x13_weighted", x13_weighted.shape)

        # 解码主路
        up = nn.Upsample(self.scale[1], mode='bilinear')
        s3 = self.skip3(x11_weighted)
        x14 = up(x13_weighted)
        x15 = self.ex9(torch.cat([s3, x14], dim=1))
        up = nn.Upsample(self.scale[0], mode='bilinear')
        s4 = self.skip4(x9_weighted)
        x16 = up(x15)
        out_fmsi = self.ex10(torch.cat([s4, x16], dim=1))
        # print("out_fmsi (decode主路输出)", out_fmsi.shape)

        # 多尺度融合（空间对齐）
        x11_weighted_up = fun.interpolate(x11_weighted, size=self.scale[0], mode='bilinear', align_corners=False)
        x13_weighted_up = fun.interpolate(x13_weighted, size=self.scale[0], mode='bilinear', align_corners=False)
        fused_fmsi = self.bandselect_fmsi([x9_weighted, x11_weighted_up, x13_weighted_up])
        # print("fused_fmsi (BandSelectBlock融合)", fused_fmsi.shape)
        # 自动初始化1x1卷积
        if self.fused_proj_fmsi is None:
            in_ch = fused_fmsi.shape[1]
            out_ch = out_fmsi.shape[1]
            self.fused_proj_fmsi = nn.Conv2d(in_ch, out_ch, 1).to(fused_fmsi.device)
            # print(f"自动初始化fused_proj_fmsi: Conv2d({in_ch}, {out_ch}, 1)")
        fused_proj_fmsi = self.fused_proj_fmsi(fused_fmsi)

        if self.out_fmsi_conv is None:
            out_ch = out_fmsi.shape[1]
            self.out_fmsi_conv = nn.Conv2d(out_ch, out_ch, 1).to(out_fmsi.device)
            # print(f"自动初始化out_fmsi_conv: Conv2d({out_ch}, {out_ch}, 1)")
        Out_fmsi = self.out_fmsi_conv(fused_proj_fmsi + out_fmsi)
        # print("Out_fmsi (最终输出)", Out_fmsi.shape)

        return Out_fhsi, Out_fmsi

########################## double_U_net_skip############################



if __name__ == "__main__":
    pass

    