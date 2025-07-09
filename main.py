# -*- coding: utf-8 -*-

"""
❗❗❗❗❗❗#主文件
"""

import torch
import time
import numpy as np
import random
from model.config import args
from model.unmix_module import SimpleUnmixNet

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

# ====================可视化===========================
import matplotlib.pyplot as plt
# 将 tensor 转换为 numpy 数组并调整维度
lr_msi_fhsi_est_numpy = lr_msi_fhsi_est.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
lr_msi_fmsi_est_numpy = lr_msi_fmsi_est.data.cpu().detach().numpy()[0].transpose(1, 2, 0)

# 可视化
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(lr_msi_fhsi_est_numpy[:,:,0])  # 这里假设取第一个波段进行可视化
plt.title('lr_msi(lr_hsi光谱退化)')
plt.subplot(1, 2, 2)
plt.imshow(lr_msi_fmsi_est_numpy[:,:,0])
plt.title('lr_msi(hr_msi空间退化)')
plt.show()

psf = blind.model.psf.data.cpu().detach().numpy()[0,0,:,:] #15 x 15  numpy
srf = blind.model.srf.data.cpu().detach().numpy()[:,:,0,0].T #46 x 8   numpy
psf_gt=blind.psf_gt #15 x 15   numpy
srf_gt=blind.srf_gt  #46 x 8   numpy

end = time.perf_counter()   # 记录结束时间
elapsed_S1 = end - start        # 计算经过的时间（单位为秒）

# # 【1】提取原始输入
# LrHSI_input = blind.tensor_lr_hsi.clone().detach()  # shape: [1, C1, H1, W1]
# HrMSI_input = blind.tensor_hr_msi.clone().detach()  # shape: [1, C2, H2, W2]
#
# # 【2】对原始输入做解混，获得端元/丰度
# n_end = 4   # 端元数，可自定义
# unmix_lr = SimpleUnmixNet(LrHSI_input.shape[1], n_end).to(LrHSI_input.device)
# unmix_hr = SimpleUnmixNet(HrMSI_input.shape[1], n_end).to(HrMSI_input.device)
#
# A_Lr, E_Lr = unmix_lr(LrHSI_input)   # [1, n_end, H1, W1], [1, C1, n_end]
# A_Hr, E_Hr = unmix_hr(HrMSI_input)   # [1, n_end, H2, W2], [1, C2, n_end]


'''第二阶段'''
# from model.spectral_up import spectral_SR
#
# spectral_sr=spectral_SR(args,lr_msi_fhsi_est.clone().detach(), lr_msi_fmsi_est.clone().detach(),  #lr_msi_fhsi_est, lr_msi_fmsi_est
#                            blind.tensor_lr_hsi,blind.tensor_hr_msi,blind.gt)
# start = time.perf_counter() # 记录开始时间
# Out_fused_s2 =spectral_sr.train() #返回的是四维tensor device上
# end = time.perf_counter()   # 记录结束时间
# elapsed_S2 = end - start        # 计算经过的时间（单位为秒）

from model.spatial_up import spatial_SR
spatial_sr = spatial_SR(args, lr_msi_fhsi_est.clone().detach(), lr_msi_fmsi_est.clone().detach(),blind.tensor_lr_hsi,blind.tensor_hr_msi,blind.gt,blind.srf_gt) #, lr_hsi, hr_msi)
start = time.perf_counter() # 记录开始时间
hr_hsi1, hr_hsi2 = spatial_sr.train()
end = time.perf_counter()   # 记录结束时间
elapsed_S2 = end - start        # 计算经过的时间（单位为秒）




'''第三阶段''' 
''' 第三阶段耗时最多，如果你觉得代码运行过慢，可以将config.py里的--band参数调小'''

from model.dip import dip #my_dip  ,  dip 
# DIP=dip(args,Out_fused_s2 ,Out_fused_s2 ,psf,srf,blind)
DIP=dip(args,hr_hsi1 ,hr_hsi2 ,psf,srf,blind)
# 【3】把物理先验传给dip
# DIP = dip(
#     args,
#     Out_fused_s2, Out_fused_s2,
#     psf, srf, blind,
#     abundance_lr=A_Lr, endmember_lr=E_Lr, abundance_hr=A_Hr, endmember_hr=E_Hr  # 新增
# )
print(DIP.net)
start = time.perf_counter() # 记录开始时间  
out_fhsi_s3,out_fmsi_s3=DIP.train() # 返回的是四维tensor device上
end = time.perf_counter()   # 记录结束时间
elapsed_S3 = end - start        # 计算经过的时间（单位为秒） 

'''第四阶段'''
from model.select import select_decision
start = time.perf_counter() # 记录开始时间  
srf_out=select_decision(out_fhsi_s3,out_fmsi_s3,blind) #返回的是3维H W C numpy 这个就是最终的结果

end = time.perf_counter()   # 记录结束时间
elapsed_S4 = end - start        # 计算经过的时间（单位为秒）

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

print(get_parameter_number(blind.model))
print(get_parameter_number(spatial_sr.net))
print(get_parameter_number(DIP.net))

print("training time S1:{},s2:{},s3:{},s4:{}".format(elapsed_S1,elapsed_S2,elapsed_S3,elapsed_S4))









