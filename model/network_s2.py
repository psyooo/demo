# -*- coding: utf-8 -*-
"""
❗❗❗❗❗❗李嘉鑫  作者微信 BatAug
空天信息创新研究院20-25直博生，导师高连如

"""
"""
❗❗❗❗❗❗#此py作用：第二阶段所需要的网络模块
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
    net.to(device)  # gpu_ids[0] 是 gpu_ids列表里面的第一个int值
    if initializer :
        # print(2,initializer)
        init_weights(net,init_type, init_gain)
    else:
        print('Spectral_downsample with default initialize')
    return net

    


#########################  def_two_stream_interactive双流交叉光谱学习网络 #########################


def def_two_stream_interactive(msi_channels,hsi_channels,device,init_type='kaiming', init_gain=0.02,initializer=True):

    net = two_stream_interactive(msi_channels,hsi_channels)
    
    return init_net(net, device, init_type, init_gain ,initializer)


class two_stream_interactive(nn.Module):
    def __init__(self,msi_channels,hsi_channels,need_clamp=False):
        super().__init__()
        self.layers=[]  # 空列表，用来存放网络层
        self.need_clamp=need_clamp  # 是否需要对输出进行clamp操作
        self.num_ups=int(np.log2(hsi_channels/msi_channels))    # 计算需要多少次上采样：log2（C/c）
        self.lrhsi_stream=nn.ModuleList([])  # 低分光谱流
        self.hrmsi_stream=nn.ModuleList([])  # 高分光谱流

        self.lrhsi_new_stream = nn.ModuleList([])
        self.hrmsi_new_stream = nn.ModuleList([])
        
        for i in range(1,self.num_ups+1): # 1 2 3 4
            feature_dim = msi_channels * (2 ** (i - 1))
            self.lrhsi_stream.append(spe(feature_dim,msi_channels*(2**(i))))
                                     
            self.hrmsi_stream.append(spe(feature_dim,msi_channels*(2**(i))))



                                     
        self.lrhsi_stream.append(nn.Conv2d(msi_channels*(2**self.num_ups), hsi_channels, kernel_size=1, stride=1, padding=0))     # 最后一层卷积层
        
        self.hrmsi_stream.append(nn.Conv2d(msi_channels*(2**self.num_ups), hsi_channels, kernel_size=1, stride=1, padding=0))     # 最后一层卷积层


    def forward(self,lrmsi_flrhsi,lrmsi_fhrmsi):
        for i in range(1,self.num_ups+1):

            lrmsi_flrhsi = self.lrhsi_stream[i-1](lrmsi_flrhsi)  # 低分光谱流K1
            lrmsi_fhrmsi = self.hrmsi_stream[i-1](lrmsi_fhrmsi)  # 高分光谱流K2
            #print(lrmsi_flrhsi.shape,lrmsi_fhrmsi.shape)

            # 低分光谱流和高分光谱流交叉相加
            lrmsi_flrhsi=lrmsi_flrhsi+lrmsi_fhrmsi  # 低分光谱流和高分光谱流相加
            lrmsi_fhrmsi=lrmsi_fhrmsi+lrmsi_flrhsi  # 高分光谱流和低分光谱流相加
            #print(lrmsi_flrhsi.shape,lrmsi_fhrmsi.shape)
        
        out_lrmsi_flrhsi=self.lrhsi_stream[-1](lrmsi_flrhsi)    # 低分光谱流输出
        out_lrmsi_fhrmsi=self.hrmsi_stream[-1](lrmsi_fhrmsi)    # 高分光谱流输出

        return out_lrmsi_flrhsi,out_lrmsi_fhrmsi
#########################  def_two_stream_interactive #########################



########################## SPE:交叉通道交互增强模块 ############################


class spe(nn.Module):
    def __init__(self,input_channel,output_channel):
        super().__init__()
    
        self.begin=nn.Sequential(
            nn.Conv2d(in_channels=input_channel,out_channels=60,kernel_size=(1,1),stride=1,padding=0) ,
            nn.LeakyReLU(0.2, inplace=True)
            )
        
        
        self.stream1= nn.Sequential(
            nn.Conv2d(in_channels=20,out_channels=20,kernel_size=(1,1),stride=1,padding=0),
            nn.LeakyReLU(0.2, inplace=True)
            )

        self.stream2=nn.Sequential(
            nn.Conv2d(in_channels=20,out_channels=20,kernel_size=(1,1),stride=1,padding=0),
            nn.LeakyReLU(0.2, inplace=True)
            )
        
        self.stream3=nn.Sequential(
            nn.Conv2d(in_channels=20,out_channels=20,kernel_size=(1,1),stride=1,padding=0),
            nn.LeakyReLU(0.2, inplace=True)
            )

        self.end=nn.Sequential(
            nn.Conv2d(in_channels=60,out_channels=output_channel,kernel_size=(1,1),stride=1,padding=0),
            nn.LeakyReLU(0.2, inplace=True)
            )
        

        
    def forward(self,input):
        
        x1=self.begin(input) # torch.Size([1, 60, 50, 50])  input:torch.Size([1, 100, 50, 50])

        # 拆分光谱波段
        split1=x1[:,0:20,:,:]   # torch.Size([1, 20, 50, 50])
        split2=x1[:,20:40,:,:]  # torch.Size([1, 20, 50, 50])
        split3=x1[:,40:,:,:]    # torch.Size([1, 20, 50, 50])

        # 中间层
        middle1=self.stream1(split1)    # torch.Size([1, 20, 50, 50])：0-20波段
        middle2=self.stream2(split2+middle1)    # torch.Size([1, 20, 50, 50])：20-40波段
        middle3=self.stream3(split3+middle2)    # torch.Size([1, 20, 50, 50])：40-60波段
        
        concat=torch.cat([ middle1, middle2,middle3 ], dim=1)   # torch.Size([1, 60, 50, 50])拼接
        
        x2=x1+concat    # torch.Size([1, 60, 50, 50])残差连接
        
        out=self.end(x2)    # torch.Size([1, 100, 50, 50])
        
        return out 

########################## SPE ############################


if __name__ == "__main__":
    import numpy as np
    import os
    import scipy.io as io

    