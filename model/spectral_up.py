# -*- coding: utf-8 -*-
"""
❗❗❗❗❗❗#此py作用：对应第二阶段的光谱上采样
"""
import torch
import torch.nn as nn
import numpy as np
from .evaluation import MetricsCal
from torch.optim import lr_scheduler
import torch.optim as optim
import os
import scipy
from datetime import datetime

from.network_s2 import def_two_stream_interactive


class spectral_SR():
    def __init__(self,args,lr_msi_fhsi,lr_msi_fmsi,lr_hsi,hr_msi,gt):
        self.args=args
        self.hs_band=lr_hsi.shape[1]  # 54    torch.Size([1, 54, 20, 20])
        self.ms_band=lr_msi_fhsi.shape[1] # 8   torch.Size([1, 8, 20, 20])
        
        self.lr_msi_fhsi=lr_msi_fhsi # torch.Size([1, 8, 20, 20]) 已经在device上的
        self.lr_msi_fmsi=lr_msi_fmsi # torch.Size([1, 8, 20, 20]) 已经在device上的
        self.lr_hsi=lr_hsi   # torch.Size([1, 54, 20, 20]) 已经在device上的    低空间分辨率的高光谱 四维tensor
        self.hr_msi=hr_msi   # torch.Size([1, 8, 240, 240]) 已经在device上的   高空间分辨率的多光谱 四维tensor
        self.gt=gt #H W C的numpy
            
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch +  1 - self.args.niter2_SPe) / float(self.args.niter_decay2_SPe + 1)
            return lr_l
        
        self.two_stream= def_two_stream_interactive(self.ms_band,self.hs_band,self.args.device)
        
        self.optimizer=optim.Adam(self.two_stream.parameters(), lr=self.args.lr_stage2_SPe) 
        self.scheduler=lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)
        

    def train(self):
        flag_best_fhsi=[10,0,'data'] #第一个是SAM，第二个是PSNR,第三个为恢复的图像
        flag_best_fmsi=[10,0,'data'] #第一个是SAM，第二个是PSNR,第三个为恢复的图像
        
        L1Loss = nn.L1Loss(reduction='mean')
        
        for epoch in range(1, self.args.niter2_SPe + self.args.niter_decay2_SPe + 1):
             
            
            self.optimizer.zero_grad()

            #输入K1、K2，输出Y1、Y2
            lr_hsi_est_fhsi,lr_hsi_est_fmsi=self.two_stream(self.lr_msi_fhsi,self.lr_msi_fmsi)    
            
            
            #计算损失：文中的L2损失：L1Loss(Y1,K1)+L1Loss(Y2,K2)
            loss=L1Loss(lr_hsi_est_fhsi, self.lr_hsi)  + L1Loss(lr_hsi_est_fmsi, self.lr_hsi)  
          
            loss.backward()
           
            self.optimizer.step()
            self.scheduler.step()
           
            if epoch % 100 ==0:

                with torch.no_grad():
                    
                    print("____________________stage-2________________________")
                    print('epoch:{} lr:{} 保存最优结果'.format(epoch,self.optimizer.param_groups[0]['lr']))
                    print('************')
                    
                    
                    #转为W H C的numpy 方便计算指标
                   
                    lr_hsi_numpy=self.lr_hsi.data.cpu().detach().numpy()[0].transpose(1,2,0)
                    lr_hsi_est_fhsi_numpy=lr_hsi_est_fhsi.data.cpu().detach().numpy()[0].transpose(1,2,0)
                    lr_hsi_est_fmsi_numpy=lr_hsi_est_fmsi.data.cpu().detach().numpy()[0].transpose(1,2,0)

                    #输入hr_msi到训练好的网络，输出gt_est_fhsi（X1）,gt_est_fmsi（X2）
                    gt_est_fhsi,gt_est_fmsi=self.two_stream(self.hr_msi,self.hr_msi)#对msi上采样到hhsi
                    
                    
                    #gt_est_fhsi_numpy=gt_est_fhsi.detach().data.cpu().numpy()[0].transpose(1,2,0) #numpy
                    #gt_est_fmsi_numpy=gt_est_fmsi.detach().data.cpu().numpy()[0].transpose(1,2,0) #numpy
                    gt_est_fhsi_numpy=gt_est_fhsi.data.cpu().numpy()[0].transpose(1,2,0).astype('float64')  #numpy dtype('float32')
                    gt_est_fmsi_numpy=gt_est_fmsi.data.cpu().numpy()[0].transpose(1,2,0).astype('float64') #numpy
                    
                 
                    #学习到的lrhsi与真值
                    sam,psnr,ergas,cc,rmse,Ssim,Uqi=MetricsCal(lr_hsi_numpy,lr_hsi_est_fhsi_numpy, self.args.scale_factor)
                    L1=np.mean( np.abs( lr_hsi_numpy - lr_hsi_est_fhsi_numpy ))
                    information1="生成lr_hsi_est_fhsi_numpy与目标lrhsi\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1,sam,psnr,ergas,cc,rmse,Ssim,Uqi)
                    print(information1) #监控训练过程
                    print('************')
                    
                    sam,psnr,ergas,cc,rmse,Ssim,Uqi=MetricsCal(lr_hsi_numpy,lr_hsi_est_fmsi_numpy, self.args.scale_factor)
                    L1=np.mean( np.abs( lr_hsi_numpy - lr_hsi_est_fmsi_numpy ))
                    information2="生成lr_hsi_est_fmsi_numpy与目标lrhsi\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1,sam,psnr,ergas,cc,rmse,Ssim,Uqi)
                    print(information2) #监控训练过程
                    print('************')
                
                    
                    #学习到的gt与真值  gt_est_fhsi
                    sam,psnr,ergas,cc,rmse,Ssim,Uqi=MetricsCal(self.gt,gt_est_fhsi_numpy, self.args.scale_factor)
                    L1=np.mean( np.abs( self.gt - gt_est_fhsi_numpy ))
                    information3="生成gt_est_fhsi_numpy与目标gt\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1,sam,psnr,ergas,cc,rmse,Ssim,Uqi)
                    print(information3)
                    print('************')
                    
                    if sam < flag_best_fhsi[0] and psnr > flag_best_fhsi[1]:        
                        flag_best_fhsi[0]=sam
                        flag_best_fhsi[1]=psnr
                        flag_best_fhsi[2]=gt_est_fhsi #保存四维tensor
    
                        information_a_fhsi=information1
                        information_b_fhsi=information3
                    
                    #学习到的gt与真值  gt_est_fmsi
                    sam,psnr,ergas,cc,rmse,Ssim,Uqi=MetricsCal(self.gt,gt_est_fmsi_numpy, self.args.scale_factor)
                    L1=np.mean( np.abs( self.gt - gt_est_fmsi_numpy ))
                    information4="生成gt_est_fmsi_numpy与目标gt\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1,sam,psnr,ergas,cc,rmse,Ssim,Uqi)
                    print(information4)
                    print('************')
                    
                    
    
    
                    if sam < flag_best_fmsi[0] and psnr > flag_best_fmsi[1]:        
                        flag_best_fmsi[0]=sam
                        flag_best_fmsi[1]=psnr
                        flag_best_fmsi[2]=gt_est_fmsi #保存四维tensor
    
                        information_a_fmsi=information2
                        information_b_fmsi=information4
                        
                        
        gt_est_fhsi_numpy=flag_best_fhsi[2].data.cpu().numpy()[0].transpose(1,2,0).astype('float64')  #numpy dtype('float32')
        gt_est_fmsi_numpy=flag_best_fmsi[2].data.cpu().numpy()[0].transpose(1,2,0).astype('float64')  #numpy

        #保存最好的结果:生成的两个HrHSI
        scipy.io.savemat(os.path.join(self.args.expr_dir, 'Out_fhsi_S2.mat'), {'Out':gt_est_fhsi_numpy})
        scipy.io.savemat(os.path.join(self.args.expr_dir, 'Out_fmsi_S2.mat'), {'Out':gt_est_fmsi_numpy})
        
        #保存精度
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
           
            
        return flag_best_fhsi[2],flag_best_fmsi[2] #返回的是四维tensor
    
if __name__ == "__main__":
    pass