# -*- coding: utf-8 -*-

from datetime import datetime

"""
❗❗❗❗❗❗#此py作用：CP分解与重构
"""
import torch
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
import torch.optim as optim
import os
import scipy
import torch.nn.functional as fun

from .evaluation import MetricsCal
from .CP_network1 import HSI_MSI_Fusion_UNet
from .visualizer import UnifiedVisualizer
import logging

# 配置日志
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
''' PSF and SRF 下采样'''
class PSF_down():
    def __call__(self, input_tensor, psf, ratio):
        _, C, _, _ = input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]  # 提取通道数
        # PSF为#1 1 ratio ratio 大小的深度卷积核：对每个光谱带独立进行空间降质
        if psf.shape[0] == 1:
            psf = psf.repeat(C, 1, 1, 1)  # 8X1X8X8
            # input_tensor: 1X8X400X400
        output_tensor = fun.conv2d(input_tensor, psf, None, (ratio, ratio),
                                   groups=C)  # psf为卷积核ratio为步长 None代表bias为0，padding默认为无
        return output_tensor


class SRF_down():
    def __call__(self, input_tensor, srf):
        output_tensor = fun.conv2d(input_tensor, srf, None)  # srf 为 (ms_band hs_bands 1 1) 的点卷积核:对每个光谱带进行线性组合
        return output_tensor


class CP_model(nn.Module):
    def __init__(self, args, hr_msi, lr_hsi, psf, srf, blind, vis):
        super().__init__()

        self.args = args
        self.rank = self.args.Rank
        # 初始化统一可视化工具，指定环境名称
        if vis is None:
            self.visualizer = UnifiedVisualizer(env_name="Model_Visualization")
        else:
            self.visualizer = vis  # 允许外部传入共享的可视化实例

        self.hr_msi = hr_msi  # 四维
        self.lr_hsi = lr_hsi  # 四维
        self.gt = blind.gt  # 三维
        self.hr_msi = blind.tensor_hr_msi  # 四维
        self.lr_hsi = blind.tensor_lr_hsi  # 四维
        self.gt = blind.gt  # 三维
        self.gt_tensor = blind.tensor_gt  # 三维

        psf_est = np.reshape(psf, newshape=(1, 1, self.args.scale_factor, self.args.scale_factor))  # 1 1 ratio ratio 大小的tensor
        self.psf_est = torch.tensor(psf_est).to(self.args.device).float()
        srf_est = np.reshape(srf.T, newshape=(srf.shape[1], srf.shape[0], 1, 1))  # self.srf.T 有一个T转置 (8, 191, 1, 1)
        self.srf_est = torch.tensor(srf_est).to(self.args.device).float()  # ms_band hs_bands 1 1 的tensor torch.Size([8, 191, 1, 1])

        self.psf_down = PSF_down()  # __call__(self, input_tensor, psf, ratio):
        self.srf_down = SRF_down()  # __call__(self, input_tensor, srf):

        self.noise1 = self.get_noise(self.gt.shape[2], (self.gt.shape[0], self.gt.shape[1])).to(self.args.device).float()
        self.noise2 = self.get_noise(self.gt.shape[2], (self.gt.shape[0], self.gt.shape[1])).to(self.args.device).float()

        self.net = HSI_MSI_Fusion_UNet(args, hr_msi.shape[1], lr_hsi.shape[1], lr_hsi.shape[1], bilinear=True, k=self.rank).to(self.args.device)

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - self.args.niter2_UNet) / float(self.args.niter_decay2_UNet + 1)
            return lr_l

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr_stage2_UNet)
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)

    def get_noise(self, input_depth, spatial_size, method='2D', noise_type='u', var=1. / 10):
        """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
        initialized in a specific way.
        Args:
            input_depth: number of channels in the tensor
            method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
            spatial_size: spatial size of the tensor to initialize
            noise_type: 'u' for uniform; 'n' for normal
            var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
        """

        def fill_noise(x, noise_type):
            """Fills tensor `x` with noise of type `noise_type`."""
            if noise_type == 'u':
                x.uniform_()
            elif noise_type == 'n':
                x.normal_()
            else:
                assert False

        if isinstance(spatial_size, int):
            spatial_size = (spatial_size, spatial_size)

        if method == '2D':
            shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        elif method == '3D':
            shape = [1, 1, input_depth, spatial_size[0], spatial_size[1]]
        else:
            assert False

        net_input = torch.zeros(shape)

        fill_noise(net_input, noise_type)
        net_input *= var

        return net_input

    def train(self):
        flag_best_1 = [50, 0, 'data', 0]  # 第一个是SAM，第二个是PSNR,第三个为恢复的图像,第四个是保存最佳结果对应的epoch
        flag_best_2 = [10, 0, 'data', 0]

        L1Loss = nn.L1Loss(reduction='mean')

        for epoch in range(1, self.args.niter2_UNet + self.args.niter_decay2_UNet + 1):

            self.optimizer.zero_grad()  # 清空梯度

            self.cp_recon, self.fused_feat = self.net(self.hr_msi, self.lr_hsi)

            ''' 生成估计的hr_msi '''
            self.hr_msi_fCP = self.srf_down(self.cp_recon, self.srf_est)  # 光谱退化
            self.hr_msi_fused = self.srf_down(self.fused_feat, self.srf_est)

            ''' 生成估计的lr_hsi '''
            self.lr_hsi_fCP = self.psf_down(self.cp_recon, self.psf_est, self.args.scale_factor)  # 空间退化
            self.lr_hsi_fused = self.psf_down(self.fused_feat, self.psf_est, self.args.scale_factor)


            # 融合图像空间退化--->lr_hsi_fCP，融合图像光谱退化--->hr_msi_fused
            loss_CP = L1Loss(self.hr_msi, self.hr_msi_fCP) + L1Loss(self.lr_hsi,self.lr_hsi_fCP)  # 空间退化+光谱退化
            loss_fused = L1Loss(self.hr_msi, self.hr_msi_fused) + L1Loss(self.lr_hsi, self.lr_hsi_fused)  # 融合特征图

            loss = loss_CP + loss_fused

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if epoch % 50 == 0:
                with torch.no_grad():
                    print("___________________stage-2__________________________")
                    print('epoch:{} lr:{}'.format(epoch, self.optimizer.param_groups[0]['lr']))
                    print('************')

                    # 转为W H C的numpy 方便计算指标
                    # hrmsi
                    hr_msi_np = self.hr_msi.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
                    hr_msi_fCP_np = self.hr_msi_fCP.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
                    hr_msi_fused_np = self.hr_msi_fused.data.cpu().detach().numpy()[0].transpose(1, 2, 0)

                    # lrhsi
                    lr_hsi_np = self.lr_hsi.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
                    lr_hsi_fCP_np = self.lr_hsi_fCP.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
                    lr_hsi_fused_np = self.lr_hsi_fused.data.cpu().detach().numpy()[0].transpose(1, 2, 0)

                    # hrhsi
                    hrhsi_est_fCP = self.cp_recon.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
                    hrhsi_est_fused = self.fused_feat.data.cpu().detach().numpy()[0].transpose(1, 2, 0)

                    # 计算指标
                    current_lr = self.optimizer.param_groups[0]['lr']

                    ''' for ftrans'''

                    # 学习到的lrhsi与真值
                    sam, psnr, ergas, cc, rmse, Ssim, Uqi = MetricsCal(lr_hsi_np, lr_hsi_fCP_np, self.args.scale_factor)
                    L1 = np.mean(np.abs(lr_hsi_np - lr_hsi_fCP_np))
                    information1 = "生成lrhsi_fCP与目标lrhsi\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1, sam, psnr, ergas, cc, rmse, Ssim, Uqi)
                    print(information1)  # 监控训练过程
                    print('************')

                    # 学习到的hrmsi与真值
                    sam, psnr, ergas, cc, rmse, Ssim, Uqi = MetricsCal(hr_msi_np, hr_msi_fCP_np,self.args.scale_factor)
                    L1 = np.mean(np.abs(hr_msi_np - hr_msi_fCP_np))
                    information2 = "生成hrmsi_fCP与目标hrmsi\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1, sam, psnr, ergas, cc, rmse, Ssim, Uqi)
                    print(information2)  # 监控训练过程
                    print('************')

                    # 学习到的gt与真值
                    sam_fCP, psnr_fCP, ergas, cc, rmse, Ssim, Uqi = MetricsCal(self.gt, hrhsi_est_fCP,self.args.scale_factor)
                    L1 = np.mean(np.abs(self.gt - hrhsi_est_fCP))
                    information3 = "生成hrhsi_est_fcp与目标hrhsi\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1, sam_fCP, psnr_fCP, ergas, cc, rmse, Ssim, Uqi)
                    print(information3)  # 监控训练过程
                    print('************')

                    file_name = os.path.join(self.args.expr_dir, 'Stage2.txt')
                    with open(file_name, 'a') as opt_file:
                        now = datetime.now().strftime("%c")
                        opt_file.write('================ Precision Log (%s) ================\n' % now)  # 精度日志文件头部
                        opt_file.write('——————————————————epoch:{}——————————————————'.format(epoch))
                        opt_file.write('\n')
                        opt_file.write(information1)
                        opt_file.write('\n')
                        opt_file.write(information2)
                        opt_file.write('\n')
                        opt_file.write(information3)
                        opt_file.write('\n')

                    if sam < flag_best_1[0] and psnr > flag_best_1[1]:
                        flag_best_1[0] = sam
                        flag_best_1[1] = psnr
                        flag_best_1[2] = self.cp_recon  # 保存四维tensor
                        flag_best_1[3] = epoch

                        information_a = information1
                        information_b = information2
                        information_c = information3

                    ''' for ftrans'''

                    print('--------------------------------')

                    ''' for funet'''
                    # 学习到的lrhsi与真值
                    sam, psnr, ergas, cc, rmse, Ssim, Uqi = MetricsCal(lr_hsi_np, lr_hsi_fused_np,self.args.scale_factor)
                    L1 = np.mean(np.abs(lr_hsi_np - lr_hsi_fused_np))
                    information1 = "生成lrhsi_fused与目标lrhsi\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1, sam, psnr, ergas, cc, rmse, Ssim, Uqi)
                    print(information1)  # 监控训练过程
                    print('************')

                    # 学习到的hrmsi与真值
                    sam, psnr, ergas, cc, rmse, Ssim, Uqi = MetricsCal(hr_msi_np, hr_msi_fused_np,self.args.scale_factor)
                    L1 = np.mean(np.abs(hr_msi_np - hr_msi_fused_np))
                    information2 = "生成hrmsi_fused与目标hrmsi\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1, sam, psnr, ergas, cc, rmse, Ssim, Uqi)
                    print(information2)  # 监控训练过程
                    print('************')

                    # 学习到的gt与真值
                    sam_fused, psnr_fused, ergas, cc, rmse, Ssim, Uqi = MetricsCal(self.gt, hrhsi_est_fused,self.args.scale_factor)
                    L1 = np.mean(np.abs(self.gt - hrhsi_est_fused))
                    information3 = "生成hrhsi_est_fused与目标hrhsi\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1, sam_fused, psnr_fused, ergas, cc, rmse, Ssim, Uqi)
                    print(information3)  # 监控训练过程
                    print('************')

                    print('————————————————————————————————')

                    file_name = os.path.join(self.args.expr_dir, 'Stage2.txt')
                    with open(file_name, 'a') as opt_file:

                        # opt_file.write('epoch:{}'.format(epoch))
                        opt_file.write('-------------------------------')
                        opt_file.write('\n')
                        opt_file.write(information1)
                        opt_file.write('\n')
                        opt_file.write(information2)
                        opt_file.write('\n')
                        opt_file.write(information3)
                        opt_file.write('\n')
                        opt_file.write('————————————————————————————————')
                    # print(type(flag_best_1[2]))
                    # print(flag_best_1[2])
                    if sam < flag_best_2[0] and psnr > flag_best_2[1]:
                        flag_best_2[0] = sam
                        flag_best_2[1] = psnr
                        flag_best_2[2] = self.fused_feat  # 保存四维tensor
                        flag_best_2[3] = epoch

                        information_d = information1
                        information_e = information2
                        information_f = information3
                    ''' for fmsi'''

                    # 使用可视化工具类更新所有可视化内容
                    self.visualizer.update_losses(epoch, loss.item(), loss_CP.item(), loss_fused.item())
                    self.visualizer.update_metrics('fCP', epoch, psnr_fCP, sam_fCP)
                    self.visualizer.update_metrics('fused', epoch, psnr_fused, sam_fused)
                    self.visualizer.update_learning_rate(epoch, current_lr)
                    self.visualizer.update_reconstructions('fCP', epoch, hrhsi_est_fCP)
                    self.visualizer.update_reconstructions('fused', epoch, hrhsi_est_fused)
                    self.visualizer.update_hrmsi(epoch, hr_msi_fCP_np, hr_msi_fused_np)
                    self.visualizer.update_lrhsi(epoch, lr_hsi_fCP_np, lr_hsi_fused_np)

                    # 光谱曲线对比
                    h, w = hrhsi_est_fCP.shape[:2]
                    x, y = np.random.randint(0, h), np.random.randint(0, w)
                    gt_curve = self.gt[x, y, :]
                    fCP_curve = hrhsi_est_fCP[x, y, :]
                    fused_curve = hrhsi_est_fused[x, y, :]
                    self.visualizer.update_spectral_curve(epoch, gt_curve, fCP_curve, fused_curve)

        # 保存最好的结果
        # scipy.io.savemat(os.path.join(self.args.expr_dir, 'Out_fhsi_S3.mat'), {'Out':flag_best_1[2].data.cpu().numpy()[0].transpose(1,2,0)})
        # scipy.io.savemat(os.path.join(self.args.expr_dir, 'Out_fmsi_S3.mat'), {'Out':flag_best_2[2].data.cpu().numpy()[0].transpose(1,2,0)})

        if isinstance(flag_best_1[2], torch.Tensor):
            scipy.io.savemat(
                os.path.join(self.args.expr_dir, 'hrhsi_1_S2.mat'),
                {'Out': flag_best_1[2].data.cpu().numpy()[0].transpose(1, 2, 0)}
            )
        else:
            print("Warning: flag_best_1[2] is not a tensor, skipping save.")
        if isinstance(flag_best_2[2], torch.Tensor):
            scipy.io.savemat(
                os.path.join(self.args.expr_dir, 'hrhsi_2_S2.mat'),
                {'Out': flag_best_2[2].data.cpu().numpy()[0].transpose(1, 2, 0)}
            )
        else:
            print("Warning: flag_best_2[2] is not a tensor, skipping save.")
        # 保存精度
        file_name = os.path.join(self.args.expr_dir, 'Stage2.txt')
        with open(file_name, 'a') as opt_file:

            opt_file.write('————————————最终结果————————————')
            opt_file.write('\n')
            opt_file.write('epoch_fhsi_best:{}'.format(flag_best_1[3]))
            opt_file.write('\n')
            opt_file.write(information_a)
            opt_file.write('\n')
            opt_file.write(information_b)
            opt_file.write('\n')
            opt_file.write(information_c)
            opt_file.write('\n')
            opt_file.write('epoch_fmsi_best:{}'.format(flag_best_2[3]))
            opt_file.write('\n')
            opt_file.write(information_d)
            opt_file.write('\n')
            opt_file.write(information_e)
            opt_file.write('\n')
            opt_file.write(information_f)

        return flag_best_1[2], flag_best_2[2]


if __name__ == "__main__":
    pass



