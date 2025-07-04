# -*- coding: utf-8 -*-
"""
❗❗❗❗❗❗李嘉鑫 作者微信 BatAug
空天信息创新研究院20-25直博生，导师高连如

"""


"""
❗❗第一阶段：SRF和PSF的生成
#此py作用：估计PSF和SRF 同时生成Lrmsi
"""
import numpy as np
import scipy.io as sio
import os
import torch
import torch.nn.functional as fun
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

from .read_data import readdata
from .evaluation import MetricsCal
import random
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from datetime import datetime



class BlurDown(object):
    def __init__(self):
        #self.shift_h = shift_h
        #self.shift_w = shift_w
        #self.stride = stride
        pass

    def __call__(self, input_tensor: torch.Tensor, psf, groups, ratio):
        if psf.shape[0] == 1:
            psf = psf.repeat(groups, 1, 1, 1) #8X1X8X8
       
        output_tensor = fun.conv2d(input_tensor, psf, None, (ratio, ratio),  groups=groups) #ratio为步长 None代表bias为0，padding默认为无
        return output_tensor

# 定义网络结构
class BlindNet(nn.Module):
    def __init__(self, hs_bands, ms_bands, ker_size, ratio):
        super().__init__()
        self.hs_bands = hs_bands    # 高光谱波段数
        self.ms_bands = ms_bands    # 多光谱波段数
        self.ker_size = ker_size # 8
        self.ratio = ratio # 8
        
        # psf = torch.rand([1, 1, self.ker_size, self.ker_size]) # 0-1均匀分布
        psf = torch.ones([1, 1, self.ker_size, self.ker_size]) * (1.0 / (self.ker_size ** 2))   # 均匀分布:所有元素乘以1/(ker_size^2)
        self.psf = nn.Parameter(psf)    # 随机初始化PSF
        
        # srf = torch.rand([self.ms_bands, self.hs_bands, 1, 1]) # 0-1均匀分布
        srf = torch.ones([self.ms_bands, self.hs_bands, 1, 1]) * (1.0 / self.hs_bands)   # 均匀分布:所有元素乘以1/hs_bands
        self.srf = nn.Parameter(srf)
        self.blur_down = BlurDown()

    def forward(self, lr_hsi, hr_msi):
        
        srf_div = torch.sum(self.srf, dim=1, keepdim=True) # 8 x 1x 1 x 1:沿波长维度（列方向）求和
        # print('srf_div',srf_div,srf_div.shape)
        
        srf_div = torch.div(1.0, srf_div)     # 8 x 1x 1 x 1:# 逐元素取倒数
        # print('srf_div',srf_div,srf_div.shape)
        
        srf_div = torch.transpose(srf_div, 0, 1)  # 1 x l x 1 x 1    1 x 8 x 1 x 1
        # print('srf_div',srf_div,srf_div.shape)

        # 光谱退化：Y(lr_hsi)----->K1(lr_msi_fhsi)
        lr_msi_fhsi = fun.conv2d(lr_hsi, self.srf, None) # (1,8,30, 30)
        lr_msi_fhsi = torch.mul(lr_msi_fhsi, srf_div) # element-wise broadcast Ylow:1X8X30X30
        lr_msi_fhsi = torch.clamp(lr_msi_fhsi, 0.0, 1.0)    # 限制在0-1之间

        #空间退化：Z(hr_msi)----->K2(lr_msi_fmsi)
        lr_msi_fmsi = self.blur_down(hr_msi, self.psf,  self.ms_bands, self.ratio)
        lr_msi_fmsi = torch.clamp(lr_msi_fmsi, 0.0, 1.0)
        return lr_msi_fhsi, lr_msi_fmsi


class Blind(readdata):
    def __init__(self, args):
        super().__init__(args)
        #self.strBR = 'BR.mat'
        # set
        self.S1_lr = self.args.lr_stage1
        self.ker_size = self.args.scale_factor  #
        self.ratio    = self.args.scale_factor 
        self.hs_bands = self.srf_gt.shape[0]
        self.ms_bands = self.srf_gt.shape[1]
        # variable, graph and etc.
        #self.__hsi = torch.tensor(self.hsi)
        #self.__msi = torch.tensor(self.msi)
        self.model = BlindNet(self.hs_bands, self.ms_bands, self.ker_size, self.ratio).to(self.args.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.S1_lr)
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch +  1 - args.niter1) / float(args.niter_decay1 + 1)
            return lr_l
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)
        

    def train(self):
        
        #hsi, msi = self.__hsi.cuda(), self.__msi.cuda()
        lr_hsi, hr_msi = self.tensor_lr_hsi.to(self.args.device), self.tensor_hr_msi.to(self.args.device)
        #for epoch in range(1, max_iter+1):
        for epoch in range(1, self.args.niter1 + self.args.niter_decay1 + 1):
            
            self.optimizer.zero_grad()
            # K1、K2生成
            lr_msi_fhsi_est, lr_msi_fmsi_est = self.model(lr_hsi, hr_msi)
            loss = torch.sum(torch.abs(lr_msi_fhsi_est - lr_msi_fmsi_est))  #文中L1损失：L1(K1-K2)
            loss.backward()
                        
            self.optimizer.step()
            self.scheduler.step()
                    
            self.model.apply(self.check_weight)
         
            if (epoch ) % 100 == 0:
                with torch.no_grad():
                    
                        print("___________________stage-1_________________________")
                        #print('epoch: %s, lr: %s, loss: %s' % (epoch, self.S1_lr, loss))
                        print('epoch:{} lr:{}'.format(epoch,self.optimizer.param_groups[0]['lr']))
                        print('************')

                        #将输出的两个LrMSI转为HWC的numpy，便于指标计算
                        lr_msi_fhsi_est_numpy=lr_msi_fhsi_est.data.cpu().detach().numpy()[0].transpose(1,2,0)
                        lr_msi_fmsi_est_numpy=lr_msi_fmsi_est.data.cpu().detach().numpy()[0].transpose(1,2,0)
                        
                      
                        
                        ######计算生成的两个LrMSI之间的误差######
                        lr=self.optimizer.param_groups[0]['lr']
                        sam,psnr,ergas,cc,rmse,Ssim,Uqi=MetricsCal(lr_msi_fhsi_est_numpy,lr_msi_fmsi_est_numpy, self.args.scale_factor)
                        L1=np.mean( np.abs( lr_msi_fhsi_est_numpy- lr_msi_fmsi_est_numpy ))
                        information1="生成的两个图像\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1,sam,psnr,ergas,cc,rmse,Ssim,Uqi)
                        print(information1)
                        
                        
                        ######计算从SRF生成的LrMSI与真值差距######
                        print('************')
                        sam,psnr,ergas,cc,rmse,Ssim,Uqi=MetricsCal(self.lr_msi_fhsi,lr_msi_fhsi_est_numpy, self.args.scale_factor)
                        L1=np.mean( np.abs( self.lr_msi_fhsi- lr_msi_fhsi_est_numpy ) )
                        information2="SRF lr_msi_fhsi_est与lr_msi_fhsi \n  L1 {} sam {},psnr {} ,ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1,sam,psnr,ergas,cc,rmse,Ssim,Uqi)
                        print(information2)
                       
                        print('************')
                        
                        ######计算从PSF生成的LrMSI与真值差距######
                        sam,psnr,ergas,cc,rmse,Ssim,Uqi=MetricsCal(self.lr_msi_fmsi,lr_msi_fmsi_est_numpy, self.args.scale_factor)
                        L1=np.mean( np.abs( self.lr_msi_fmsi- lr_msi_fmsi_est_numpy ) )
                        information3="PSF lr_msi_fmsi_est与lr_msi_fmsi\n L1 {} sam {},psnr {} ,ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1,sam,psnr,ergas,cc,rmse,Ssim,Uqi)
                        print(information3)
                        #print('************')
                       
                        
                        psf_info="estimated psf \n {} \n psf_gt \n{}".format(
                            np.squeeze(self.model.psf.data.cpu().detach().numpy()), # scale_factor X scale_factor
                            self.psf_gt
                            )
                       #可以自己输出看具体结果
                            
                            
                        srf_info="estimated srf \n {} \n srf_gt \n{}".format(
                            np.squeeze(self.model.srf.data.cpu().detach().numpy()).T, # 65X4
                            self.srf_gt 
                            )
                       #可以自己输出看具体结果
                       
                        print('************')
                        
                        #####计算PSF对GT生成LrHSI与真值的差距####
                        psf = self.model.psf.repeat(self.hs_bands, 1, 1, 1)
                        lr_hsi_est = fun.conv2d(self.tensor_gt.to(self.args.device), 
                                                psf, None, (self.ker_size, self.ker_size),  
                                                groups=self.hs_bands)

                        lr_hsi_est_numpy=lr_hsi_est.data.cpu().detach().numpy()[0].transpose(1,2,0)
                
                        sam,psnr,ergas,cc,rmse,Ssim,Uqi=MetricsCal(self.lr_hsi,lr_hsi_est_numpy, self.args.scale_factor)
                        L1=np.mean( np.abs( self.lr_hsi- lr_hsi_est_numpy ) )
                        information4="PSF lr_hsi_est与lr_hsi\n L1 {} sam {},psnr {} ,ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1,sam,psnr,ergas,cc,rmse,Ssim,Uqi)
                        print(information4)
                        
                        print('************')
              
                        #####计算SRF对GT生成HrMSI与真值的差距####
                        if self.model.srf.data.cpu().detach().numpy().shape[0]!=1:
                            srf_est=np.squeeze(self.model.srf.data.cpu().detach().numpy()).T
                        else:
                            srf_est_tmp=np.squeeze(self.model.srf.data.cpu().detach().numpy()).T
                            srf_est=srf_est_tmp[:,np.newaxis]                   
                        w,h,c = self.gt.shape
                        if srf_est.shape[0] == c:
                            hr_msi_est_numpy = np.dot(self.gt.reshape(w*h,c), srf_est).reshape(w,h,srf_est.shape[1])
                        
                        
                        sam,psnr,ergas,cc,rmse,Ssim,Uqi=MetricsCal(self.hr_msi,hr_msi_est_numpy, self.args.scale_factor)
                        L1=np.mean( np.abs( self.hr_msi- hr_msi_est_numpy ) )
                        information5="SRF hr_msi_est与hr_msi\n L1 {} sam {},psnr {} ,ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1,sam,psnr,ergas,cc,rmse,Ssim,Uqi)
                        print(information5)
                        
                        
        
        
        PATH=os.path.join(self.args.expr_dir,self.model.__class__.__name__+'.pth')
        torch.save(self.model.state_dict(),PATH)
        #self.psf = torch.tensor(torchkits.to_numpy(self.model.psf.data))
        #self.srf = torch.tensor(torchkits.to_numpy(self.model.srf.data))




        file_name = os.path.join(self.args.expr_dir, 'Stage1.txt')
        with open(file_name, 'a') as opt_file:

            now = datetime.now().strftime("%c")
            opt_file.write('================ Precision Log (%s) ================\n' % now)  # 精度日志文件头部
            opt_file.write(information1)
            opt_file.write('\n')
            opt_file.write(information2)
            opt_file.write('\n')
            opt_file.write(information3)
            opt_file.write('\n')
            opt_file.write(information4)
            opt_file.write('\n')
            opt_file.write(information5)
            opt_file.write('\n')
            
        lr_msi_fhsi_est_numpy=lr_msi_fhsi_est.data.cpu().detach().numpy()[0].transpose(1,2,0)
        lr_msi_fmsi_est_numpy=lr_msi_fmsi_est.data.cpu().detach().numpy()[0].transpose(1,2,0)
        
        #保存生成的两个lrmsi:
        # lr_msi_fhsi:由lr_hsi生成的lr_msi，用于计算PSF误差
        # lr_msi_fmsi:由hr_msi生成的lr_msi，用于计算SRF误差
        # estimated_lr_msi.mat:保存lr_msi_fhsi和lr_msi_fmsi
        sio.savemat(os.path.join(self.args.expr_dir , 'estimated_lr_msi.mat'), {'lr_msi_fhsi': lr_msi_fhsi_est_numpy, 'lr_msi_fmsi': lr_msi_fmsi_est_numpy})
        return lr_msi_fhsi_est, lr_msi_fmsi_est

    # 获取并保存估计的PSF和SRF
    def get_save_result(self):
        
        #self.model.load_state_dict(torch.load(self.model_save_path + 'parameter.pkl'))
        psf = self.model.psf.data.cpu().detach().numpy() ## 1 1 15 15
        srf = self.model.srf.data.cpu().detach().numpy() # 8 46 1 1
        psf = np.squeeze(psf)  #15 15
        srf = np.squeeze(srf).T  # b x B 8 X 46 变为 46X8 和srf_gt保持一致
        #保存估计的PSF和SRF
        sio.savemat(os.path.join(self.args.expr_dir , 'estimated_psf_srf.mat'), {'psf_est': psf, 'srf_est': srf})
       

    @staticmethod
    def check_weight(model):
        # 检查并调整PSF和SRF权重
        if hasattr(model, 'psf'):
            #print(model,'psf')
            w = model.psf.data
            w.clamp_(0.0, 1.0)
            psf_div = torch.sum(w)     #1         
            psf_div = torch.div(1.0, psf_div)   #2                                                             
            w.mul_(psf_div)        #3
        
        if hasattr(model, 'srf'):
            #print(model,'srf')
            w = model.srf.data # torch.Size([8, 46, 1, 1])        
            w.clamp_(0.0, 10.0)
            srf_div = torch.sum(w, dim=1, keepdim=True) #torch.Size([8, 1, 1, 1])
            srf_div = torch.div(1.0, srf_div) #torch.Size([8, 1, 1, 1])
            w.mul_(srf_div)
            
if __name__ == "__main__":
    pass