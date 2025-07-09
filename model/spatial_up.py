import torch
import torch.nn as nn
import numpy as np
from .evaluation import MetricsCal
from torch.optim import lr_scheduler
import torch.optim as optim
import os
import scipy
from datetime import datetime
from model.network_s2_1 import SpaceUpNet, create_inverse_PSF,MatrixDotLR2HR

def get_space_sr_net(inv_psf, scale, input_x):
    n_channels = input_x.shape[1]
    return MatrixDotLR2HR(inv_psf, scale, max_channels=n_channels)
class spatial_SR(object):
    def __init__(self, args, lr_msi_fhsi, lr_msi_fmsi, lr_hsi, hr_msi,gt,srf_gt):
        self.args = args
        self.lr_msi_fhsi = lr_msi_fhsi
        self.lr_msi_fmsi = lr_msi_fmsi
        self.lr_hsi = lr_hsi
        self.hr_msi = hr_msi
        self.gt =gt
        self.srf_gt = srf_gt
        # 直接生成逆PSF核（参数可配置）
        sigma = getattr(args, 'psf_sigma', 1.5)

        inv_psf = create_inverse_PSF(sigma, shape=(self.args.scale_factor, self.args.scale_factor), device=args.device)
        self.inv_psf = inv_psf
        self.scale_factor = args.scale_factor
        self.space_sr_net_msi = get_space_sr_net(inv_psf, self.scale_factor, self.lr_msi_fhsi).to(args.device)
        # n_channels = lr_msi_fhsi.shape[1]


        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch +  1 - self.args.niter2_SPa) / float(self.args.niter_decay2_SPa + 1)
            return lr_l

        # # 自动推算 scale
        # scale_h = hr_msi.shape[2] // lr_msi_fhsi.shape[2]
        # scale_w = hr_msi.shape[3] // lr_msi_fhsi.shape[3]
        # assert scale_h == scale_w, "空间分辨率缩放因子必须一致"
        # scale = scale_h
        # self.space_sr_net = MatrixDotLR2HR(inv_psf,self.args.scale_factor, n_channels).to(args.device)

        self.optimizer = torch.optim.Adam(self.space_sr_net_msi.parameters(), lr=args.lr_stage2_SPa)
        self.scheduler=lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)





    def train(self):
        flag_best_fhsi = [10, 0, 'data']  # 第一个是SAM，第二个是PSNR,第三个为恢复的图像
        flag_best_fmsi = [10, 0, 'data']  # 第一个是SAM，第二个是PSNR,第三个为恢复的图像

        L1Loss = nn.L1Loss(reduction='mean')

        print(f"Stage2 will train for {self.args.niter2_SPa + self.args.niter_decay2_SPa} epochs")
        for epoch in range(1, self.args.niter2_SPa + self.args.niter_decay2_SPa + 1):
            self.optimizer.zero_grad()
            space_sr_net_msi = get_space_sr_net(self.inv_psf, self.scale_factor, self.lr_msi_fhsi).to(self.args.device)
            # 两路输入分别forward
            hr_msi_est_fhsi = self.space_sr_net_msi(self.lr_msi_fhsi)
            hr_msi_est_fmsi = self.space_sr_net_msi(self.lr_msi_fmsi)

            print("hr_msi_est_fhsi.shape", hr_msi_est_fhsi.shape)
            print("self.hr_msi.shape", self.hr_msi.shape)

            # L1损失
            loss = L1Loss(hr_msi_est_fhsi, self.hr_msi) + L1Loss(hr_msi_est_fmsi, self.hr_msi)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if epoch % 100 == 0:
                with torch.no_grad():
                    print("____________________SpaceSR-Stage2________________________")
                    print('epoch:{} lr:{} 保存最优结果'.format(epoch, self.optimizer.param_groups[0]['lr']))
                    print('************')
                    hr_msi_numpy = self.hr_msi.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
                    hr_msi_est_fhsi_numpy = hr_msi_est_fhsi.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
                    hr_msi_est_fmsi_numpy = hr_msi_est_fmsi.data.cpu().detach().numpy()[0].transpose(1, 2, 0)

                    # 用训练好的网络，输入Lr-HSI做推理
                    # gt_est_fhsi = self.space_sr_net(self.lr_hsi)
                    # gt_est_fmsi = self.space_sr_net(self.lr_hsi)
                    # lr_hsi_proj = self.hsi_to_msi_proj(self.lr_hsi)
                    # gt_est_fhsi, gt_est_fmsi = self.space_sr_net(lr_hsi_proj, lr_hsi_proj)

                    # # Step1: 高光谱→多光谱投影
                    # msi_sim = self.hsi_to_msi_proj(self.lr_hsi)  # [1, 8, h, w]
                    # # Step2: 空间超分
                    # hr_hsi1, hr_hsi2 = self.space_sr_net(msi_sim, msi_sim)  # [1, 8, H, W]
                    # # Step3: 多光谱→高光谱重建
                    # gt_est_fhsi = self.msi_to_hsi_proj(hr_hsi1)  # [1, 46, H, W]
                    # gt_est_fmsi = self.msi_to_hsi_proj(hr_hsi2)
                    space_sr_net_hsi = get_space_sr_net(self.inv_psf, self.scale_factor, self.lr_hsi).to(self.args.device)
                    gt_est_fhsi = space_sr_net_hsi(self.lr_hsi)
                    gt_est_fmsi = space_sr_net_hsi(self.lr_hsi)

                    gt_est_fhsi_numpy = gt_est_fhsi.data.cpu().numpy()[0].transpose(1, 2, 0).astype('float64')  # numpy dtype('float32')
                    gt_est_fmsi_numpy = gt_est_fmsi.data.cpu().numpy()[0].transpose(1, 2, 0).astype('float64')  # numpy

                    # 学习到的hrmsi与真值
                    sam, psnr, ergas, cc, rmse, Ssim, Uqi = MetricsCal(hr_msi_numpy, hr_msi_est_fhsi_numpy,self.args.scale_factor)
                    L1 = np.mean(np.abs(hr_msi_numpy - hr_msi_est_fhsi_numpy))
                    information1 = "生成hr_msi_est_fhsi_numpy与目标hrmsi\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(
                        L1, sam, psnr, ergas, cc, rmse, Ssim, Uqi)
                    print(information1)  # 监控训练过程
                    print('************')

                    sam, psnr, ergas, cc, rmse, Ssim, Uqi = MetricsCal(hr_msi_numpy, hr_msi_est_fmsi_numpy,self.args.scale_factor)
                    L1 = np.mean(np.abs(hr_msi_numpy - hr_msi_est_fmsi_numpy))
                    information2 = "生成hr_msi_est_fmsi_numpy与目标hrmsi\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(
                        L1, sam, psnr, ergas, cc, rmse, Ssim, Uqi)
                    print(information2)  # 监控训练过程
                    print('************')

                    # 学习到的gt与真值  gt_est_fhsi
                    sam, psnr, ergas, cc, rmse, Ssim, Uqi = MetricsCal(self.gt, gt_est_fhsi_numpy,self.args.scale_factor)
                    L1 = np.mean(np.abs(self.gt - gt_est_fhsi_numpy))
                    information3 = "生成gt_est_fhsi_numpy与目标gt\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(
                        L1, sam, psnr, ergas, cc, rmse, Ssim, Uqi)
                    print(information3)
                    print('************')

                    if sam < flag_best_fhsi[0] and psnr > flag_best_fhsi[1]:
                        flag_best_fhsi[0] = sam
                        flag_best_fhsi[1] = psnr
                        flag_best_fhsi[2] = gt_est_fhsi  # 保存四维tensor

                        information_a_fhsi = information1
                        information_b_fhsi = information3

                    # 学习到的gt与真值  gt_est_fmsi
                    sam, psnr, ergas, cc, rmse, Ssim, Uqi = MetricsCal(self.gt, gt_est_fmsi_numpy,self.args.scale_factor)
                    L1 = np.mean(np.abs(self.gt - gt_est_fmsi_numpy))
                    information4 = "生成gt_est_fmsi_numpy与目标gt\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(
                        L1, sam, psnr, ergas, cc, rmse, Ssim, Uqi)
                    print(information4)
                    print('************')

                    if sam < flag_best_fmsi[0] and psnr > flag_best_fmsi[1]:
                        flag_best_fmsi[0] = sam
                        flag_best_fmsi[1] = psnr
                        flag_best_fmsi[2] = gt_est_fmsi  # 保存四维tensor

                        information_a_fmsi = information2
                        information_b_fmsi = information4

                gt_est_fhsi_numpy = flag_best_fhsi[2].data.cpu().numpy()[0].transpose(1, 2, 0).astype('float64')  # numpy dtype('float32')
                gt_est_fmsi_numpy = flag_best_fmsi[2].data.cpu().numpy()[0].transpose(1, 2, 0).astype('float64')  # numpy

                # 保存最好的结果:生成的两个HrHSI
                scipy.io.savemat(os.path.join(self.args.expr_dir, 'Out_fhsi_S2.mat'), {'Out': gt_est_fhsi_numpy})
                scipy.io.savemat(os.path.join(self.args.expr_dir, 'Out_fmsi_S2.mat'), {'Out': gt_est_fmsi_numpy})

                # 保存精度
                file_name = os.path.join(self.args.expr_dir, 'Stage2.txt')
                with open(file_name, 'a') as opt_file:
                    now = datetime.now().strftime("%c")
                    opt_file.write('================ Precision Log (%s) ================\n' % now)  # 精度日志文件头部
                    opt_file.write(information_a_fhsi)
                    opt_file.write('\n')
                    opt_file.write(information_b_fhsi)
                    opt_file.write('\n')

                    opt_file.write(information_a_fmsi)
                    opt_file.write('\n')
                    opt_file.write(information_b_fmsi)
                    opt_file.write('\n')

                # 原代码
                return flag_best_fhsi[2], flag_best_fmsi[2] #返回的是四维tensor device上

                # # 新增: CovBlock + BandSelectBlock融合
                # C = flag_best_fhsi[2].shape[1]
                # bandselect = BandSelectBlock(C, 2).to(flag_best_fhsi[2].device)
                # fused_hrhsi = bandselect([flag_best_fhsi[2], flag_best_fmsi[2]])
                #
                # # 保存融合后的输出（如需）
                # fused_hrhsi_numpy = fused_hrhsi.data.cpu().numpy()[0].transpose(1, 2, 0).astype('float64')
                # scipy.io.savemat(os.path.join(self.args.expr_dir, 'Out_fused_S2.mat'), {'Out': fused_hrhsi_numpy})
                #
                # # 修改返回，作为后续第三阶段输入
                # return fused_hrhsi, flag_best_fhsi[2], flag_best_fmsi[
                #     2]  # 只返回融合后的，也可 return fused_hrhsi, flag_best_fhsi[2], flag_best_fmsi[2]，如第三阶段还需用到原始分支

if __name__ == "__main__":
    pass

        # with torch.no_grad():
        #     hr_hsi1, hr_hsi2 = self.two_stream(self.lr_hsi, self.lr_hsi)
        # return hr_hsi1, hr_hsi2
