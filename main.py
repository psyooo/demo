# -*- coding: utf-8 -*-

"""
❗❗❗❗❗❗#主文件
"""
import os

import torch
import time
import numpy as np
import random
from model.config import args
import matplotlib.pyplot as plt
import visdom
#设置随机种子
def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True

setup_seed(args.seed)

'''第一阶段'''
from model.srf_psf_layer import Blind       #将退化函数看作一层的参数
blind = Blind(args)

start = time.perf_counter() # 记录开始时间
lr_msi_fhsi_est, lr_msi_fmsi_est=blind.train() #device格式的1 C H W张量
blind.get_save_result() #保存PSF SRF

psf = blind.model.psf.data.cpu().detach().numpy()[0,0,:,:]  # 15 x 15  numpy
srf = blind.model.srf.data.cpu().detach().numpy()[:,:,0,0].T    # 46 x 8   numpy
psf_gt = blind.psf_gt   # 15 x 15   numpy
srf_gt = blind.srf_gt  # 46 x 8   numpy
gt = blind.tensor_gt    # 1 C H W张量
lr_hsi = blind.tensor_lr_hsi    # 1 C H W张量
hr_msi = blind.tensor_hr_msi    # 1 C H W张量


end = time.perf_counter()   # 记录结束时间
elapsed_S1 = end - start        # 计算经过的时间（单位为秒）

'''第二阶段'''


from model.CP import CP_model
CP=CP_model(args,blind.tensor_hr_msi,blind.tensor_lr_hsi,psf_gt,srf_gt,blind,vis)
# print(CP.net)
start = time.perf_counter() # 记录开始时间
hr_hsi1, hr_hsi2 =CP.train() # 返回的是四维tensor device上
end = time.perf_counter()   # 记录结束时间
elapsed_S2 = end - start        # 计算经过的时间（单位为秒）


'''第三阶段'''
from model.select import select_decision
# from model.fusion import select_decision
start = time.perf_counter() # 记录开始时间
srf_out=select_decision(hr_hsi1,hr_hsi2,blind) #返回的是3维H W C numpy 这个就是最终的结果
# srf_out=Fusion.train() # 返回的是3维H W C numpy 这个就是最终的结果

end = time.perf_counter()   # 记录结束时间
elapsed_S4 = end - start        # 计算经过的时间（单位为秒）


# ====================visdom可视化===========================
hsi_bands = [45, 29, 13]  # 注意：Python索引从0开始，所以实际索引为46-1, 30-1, 14-1
msi_bands = [5, 4, 2]

# 第一阶段结果可视化
lr_msi_fhsi_est_numpy = lr_msi_fhsi_est.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
lr_msi_fmsi_est_numpy = lr_msi_fmsi_est.data.cpu().detach().numpy()[0].transpose(1, 2, 0)

# 第二阶段结果可视化
hr_hsi1_numpy = hr_hsi1.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
hr_hsi2_numpy = hr_hsi2.data.cpu().detach().numpy()[0].transpose(1, 2, 0)

# 将GT、lr_hsi、hr_msi转为numpy
gt_numpy = gt.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
lr_hsi_numpy = lr_hsi.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
hr_msi_numpy = hr_msi.data.cpu().detach().numpy()[0].transpose(1, 2, 0)


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

print(get_parameter_number(blind.model))
print(get_parameter_number(CP.net))
# print(get_parameter_number(Fusion.net))

# print("training time S1:{},s2:{},s3:{},s4:{}".format(elapsed_S1,elapsed_S2,elapsed_S3,elapsed_S4))
print("training time S1:{},s2:{},s4:{}".format(elapsed_S1,elapsed_S2,elapsed_S4))
