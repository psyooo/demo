# -*- coding: utf-8 -*-

"""
❗❗❗❗❗❗#此py作用：修改本方法所有超参数
"""

#本方法可以调整的超参数都在此.py

import argparse
import torch
import os
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
###通用参数
parser.add_argument('--scale_factor',type=int,default=8, help='缩放尺度 houston18=8 DC=10 TG=12 Chikusei=16')
parser.add_argument('--sp_root_path',type=str, default='data/A/spectral_response/',help='光谱响应地址')
parser.add_argument('--default_datapath',type=str, default="data/A/",help='高光谱数据读取地址')
parser.add_argument('--data_name',type=str, default="houston18",help='houston18=8 DC=10 TG=12 Chikusei=16')
parser.add_argument("--gpu_ids", type=str, default='0', help='指定运行的GPU ID')
parser.add_argument('--checkpoints_dir',type=str, default='checkpoint',help='结果存储地址')
parser.add_argument('--seed',type=int, default=30,help='初始化种子')

####
''' 第三阶段耗时最多，如果你觉得代码运行过慢，可以将--band参数调小'''
parser.add_argument("--band", type=int, default=200,help='设置第三阶段中U-net隐藏层特征通道数目，论文设置为240')
parser.add_argument("--select", type=str, default=True,help='是否在第4阶段采用 退化引导融合策略')

#训练参数
parser.add_argument("--lr_stage1", type=float, default=0.001,help='学习率6e-3 0.001')
parser.add_argument('--niter1', type=int, default=1000, help='# of iter at starting learning rate2000')
parser.add_argument('--niter_decay1', type=int, default=1000, help='# of iter to linearly decay learning rate to zero2000')

# parser.add_argument("--lr_stage2_SPe", type=float, default=4e-3,help='学习率4e-3')
# parser.add_argument('--niter2_SPe', type=int, default=2000, help='#of iter at starting learning rate 2000')
# parser.add_argument('--niter_decay2_SPe', type=int, default=2000, help='#of iter to linearly decay learning rate to zero')

parser.add_argument("--lr_stage2_UNet", type=float, default=4e-3,help='学习率4e-3')
parser.add_argument('--niter2_UNet', type=int, default=100, help='#of iter at starting learning rate 7500')
parser.add_argument('--niter_decay2_UNet', type=int, default=100, help='#of iter to linearly decay learning rate to zero')
parser.add_argument("--Rank", type=int, default=7,help='CP分解的秩')

parser.add_argument("--lr_stage3", type=float, default=4e-3,help='学习率4e-3')
parser.add_argument('--niter3', type=int, default=5000, help='# of iter at starting learning rate 7000')
parser.add_argument('--niter_decay3', type=int, default=5000, help='# of iter to linearly decay learning rate to zero')
# parser.add_argument('--num_filters', type=float, default=3, help='空洞卷积层数')



#添加噪声
parser.add_argument('--noise', type=str, default="No", help='是否对输入的数据添加噪声：Yes ,No')
parser.add_argument('--nSNR', type=int, default=35)


args=parser.parse_args()

device = torch.device(  'cuda:{}'.format(args.gpu_ids)  ) if  torch.cuda.is_available() else torch.device('cpu') 
args.device=device
args.sigma = args.scale_factor / 2.35482

#针对每种数据和不同配置的结果 存储到以下名称的文件夹里
args.expr_dir=os.path.join('checkpoint', args.data_name+'SF'+str(args.scale_factor)+'_CPRank'+str(args.Rank)+'_noise-'+str(args.noise)+
                           '_S1_'+str(args.lr_stage1)+'_'+str(args.niter1)+'_'+str(args.niter_decay1)+
                           '_S2_'+str(args.lr_stage2_UNet)+'_'+str(args.niter2_UNet)+'_'+str(args.niter_decay2_UNet)
                           )


