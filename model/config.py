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
parser.add_argument('--scale_factor',type=int,default=4, help='缩放尺度 houston18=8 DC=10 TG=12 Chikusei=16')
parser.add_argument('--sp_root_path',type=str, default='data/A/spectral_response/',help='光谱响应地址')
parser.add_argument('--default_datapath',type=str, default="data/A/",help='高光谱数据读取地址')
parser.add_argument('--data_name',type=str, default="houston",help='houston18=8 DC=10 TG=12 Chikusei_128=16 PaviaU_256x256、houston、Indian_pines')
parser.add_argument("--gpu_ids", type=str, default='0', help='指定运行的GPU ID')
parser.add_argument('--checkpoints_dir',type=str, default='houston',help='结果存储地址')
parser.add_argument('--seed',type=int, default=30,help='初始化种子')

####
''' 第三阶段耗时最多，如果你觉得代码运行过慢，可以将--band参数调小'''
parser.add_argument("--select", type=str, default=True,help='是否在第4阶段采用 退化引导融合策略')

#训练参数
parser.add_argument("--lr_stage1", type=float, default=0.001,help='学习率6e-3 0.001')
parser.add_argument('--niter1', type=int, default=1000, help='# of iter at starting learning rate2000')
parser.add_argument('--niter_decay1', type=int, default=1000, help='# of iter to linearly decay learning rate to zero2000')


parser.add_argument("--lr_stage2_UNet", type=float, default=4e-3,help='学习率4e-3')
parser.add_argument('--niter2_UNet', type=int, default=8500, help='#of iter at starting learning rate 7500')
parser.add_argument('--niter_decay2_UNet', type=int, default=8500, help='#of iter to linearly decay learning rate to zero')
parser.add_argument("--Rank", type=int, default=25,help='CP分解的秩')

# 在 parser.add_argument("--Rank", ...) 之后追加
# parser.add_argument("--auto_rank", action="store_true", default=True,help="在read_data阶段基于GT估计端元数/VD并回填Rank上界")
# parser.add_argument("--rank_vd_multiplier", type=float, default=1.5,help="Rank ≈ multiplier * VD 的系数")
# parser.add_argument("--rank_min", type=int, default=6,help="自动估计后允许的最小Rank（下限）")


parser.add_argument("--l21_weight", type=float, default=0,help="组稀疏 L2,1 正则：自动剪枝 CP 分量")
parser.add_argument("--lambda_l1_weight", type=float, default=0,help="对 lambda_weight 的 L1 稀疏（促使少量秩-1 分量有效）")



#添加噪声
parser.add_argument('--noise', type=str, default="No", help='是否对输入的数据添加噪声：Yes ,No')
parser.add_argument('--nSNR', type=int, default=35)

parser.add_argument('--save_weight_map', action='store_true', default=True, help='Save fusion weight map')



args=parser.parse_args()

device = torch.device(  'cuda:{}'.format(args.gpu_ids)  ) if  torch.cuda.is_available() else torch.device('cpu') 
args.device=device
args.sigma = args.scale_factor / 2.35482

#针对每种数据和不同配置的结果 存储到以下名称的文件夹里
args.expr_dir=os.path.join('houston', args.data_name+'SF'+str(args.scale_factor)+'_CPRank'+str(args.Rank)+'_noise-'+str(args.noise)+
                           '_S1_'+str(args.lr_stage1)+'_'+str(args.niter1)+'_'+str(args.niter_decay1)+
                           '_S2_'+str(args.lr_stage2_UNet)+'_'+str(args.niter2_UNet)+'_'+str(args.niter_decay2_UNet)
                           )


