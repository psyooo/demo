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
import matplotlib.pyplot as plt
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


# ====================可视化===========================
hsi_bands = [45, 29, 13]  # 注意：Python索引从0开始，所以实际索引为46-1, 30-1, 14-1
msi_bands = [7, 5, 3]

# 第一阶段结果可视化
lr_msi_fhsi_est_numpy = lr_msi_fhsi_est.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
lr_msi_fmsi_est_numpy = lr_msi_fmsi_est.data.cpu().detach().numpy()[0].transpose(1, 2, 0)

# 第二阶段结果可视化
hr_hsi1_numpy = hr_hsi1.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
hr_hsi2_numpy = hr_hsi2.data.cpu().detach().numpy()[0].transpose(1, 2, 0)

# 第三阶段结果可视化
out_fhsi_s3_numpy = out_fhsi_s3.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
out_fmsi_s3_numpy = out_fmsi_s3.data.cpu().detach().numpy()[0].transpose(1, 2, 0)

# 可视化每个阶段的结果
plt.figure(figsize=(15, 10))

# 第一阶段结果可视化
plt.subplot(3, 2, 1)
plt.imshow(lr_msi_fhsi_est_numpy[:, :, msi_bands])
plt.title('Stage 1: lr_msi(lr_hsi光谱退化)')

plt.subplot(3, 2, 2)
plt.imshow(lr_msi_fmsi_est_numpy[:, :, msi_bands])
plt.title('Stage 1: lr_msi(hr_msi空间退化)')

# 第二阶段结果可视化
hr_hsi1_rgb = hr_hsi1_numpy[:, :, hsi_bands]
hr_hsi2_rgb = hr_hsi2_numpy[:, :, hsi_bands]

plt.subplot(3, 2, 3)
plt.imshow(hr_hsi1_rgb)
plt.title('Stage 2: hr_hsi1')

plt.subplot(3, 2, 4)
plt.imshow(hr_hsi2_rgb)
plt.title('Stage 2: hr_hsi2')

# 第三阶段结果可视化
out_fhsi_s3_rgb = out_fhsi_s3_numpy[:, :, hsi_bands]
out_fmsi_s3_rgb = out_fmsi_s3_numpy[:, :, hsi_bands]

plt.subplot(3, 2, 5)
plt.imshow(out_fhsi_s3_rgb)
plt.title('Stage 3: out_fhsi_s3')

plt.subplot(3, 2, 6)
plt.imshow(out_fmsi_s3_rgb)
plt.title('Stage 3: out_fmsi_s3')

plt.tight_layout()
plt.show()

# 计算最终融合图像与gt图的RMSE热图、SAM热图和重建误差图
def compute_rmse_map(gt, pred):
    rmse_map = np.sqrt(np.mean((gt - pred) ** 2, axis=-1))
    return rmse_map

def compute_sam_map(gt, pred):
    num_pixels = gt.shape[0] * gt.shape[1]
    gt_flat = gt.reshape(num_pixels, -1)
    pred_flat = pred.reshape(num_pixels, -1)
    dot_product = np.sum(gt_flat * pred_flat, axis=1)
    norm_gt = np.linalg.norm(gt_flat, axis=1)
    norm_pred = np.linalg.norm(pred_flat, axis=1)
    cos_theta = dot_product / (norm_gt * norm_pred)
    sam_map = np.arccos(cos_theta).reshape(gt.shape[:2])
    return sam_map

rmse_map = compute_rmse_map(gt, srf_out)
sam_map = compute_sam_map(gt, srf_out)
reconstruction_error_map = np.abs(gt - srf_out)

# 可视化RMSE热图、SAM热图和重建误差图
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(rmse_map, cmap='hot')
plt.title('RMSE Map')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(sam_map, cmap='hot')
plt.title('SAM Map')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(reconstruction_error_map[:, :, 0], cmap='hot')
plt.title('Reconstruction Error Map')
plt.colorbar()

plt.tight_layout()
plt.show()

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

print(get_parameter_number(blind.model))
print(get_parameter_number(spatial_sr.space_sr_net_msi))
print(get_parameter_number(DIP.net))

print("training time S1:{},s2:{},s3:{},s4:{}".format(elapsed_S1,elapsed_S2,elapsed_S3,elapsed_S4))









