# -*- coding: utf-8 -*-

from datetime import datetime

"""
❗❗❗❗❗❗#此py作用：对应第三阶段的图像生成
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
from .network_s3_1 import DualBranchAsymNet


from .network_s3 import double_U_net_skip
from model.network_Tucker import MultiStreamHierarchicalTuckerNet

def tucker_rank_reg(model, lambda_rank=1e-2):
    reg = 0.
    for block in [model.block1, model.block2]:
        for g in [block.gate1, block.gate2, block.gate3, block.gateG]:
            reg = reg + torch.sigmoid(g).abs().sum()
    return lambda_rank * reg

''' PSF and SRF 下采样'''
class PSF_down():
    def __call__(self, input_tensor, psf, ratio):
            _,C,_,_=input_tensor.shape[0],input_tensor.shape[1],input_tensor.shape[2],input_tensor.shape[3] # 提取通道数
            # PSF为#1 1 ratio ratio 大小的深度卷积核：对每个光谱带独立进行空间降质
            if psf.shape[0] == 1:
                psf = psf.repeat(C, 1, 1, 1) #8X1X8X8
                                                   #input_tensor: 1X8X400X400
            output_tensor = fun.conv2d(input_tensor, psf, None, (ratio, ratio),  groups=C) # psf为卷积核ratio为步长 None代表bias为0，padding默认为无
            return output_tensor
    
class SRF_down(): 
    def __call__(self, input_tensor, srf):
            output_tensor = fun.conv2d(input_tensor, srf, None) # srf 为 (ms_band hs_bands 1 1) 的点卷积核:对每个光谱带进行线性组合
            return output_tensor
        
class dip():
    # def __init__(self, args, hrhsi_input1, hrhsi_input2, psf, srf, blind,
    #              abundance_lr=None, endmember_lr=None,
    #              abundance_hr=None, endmember_hr=None):
    def __init__(self,args,hrhsi_input1,hrhsi_input2,psf,srf,blind):
        super().__init__()
        # assert(hrhsi_input1.shape == hrhsi_input2.shape)
        
        #获取SRF and PSF
        
        self.hrhsi_input1=hrhsi_input1
        self.hrhsi_input2=hrhsi_input2
        
        self.args=args
        
        self.hr_msi=blind.tensor_hr_msi #四维
        self.lr_hsi=blind.tensor_lr_hsi #四维
        self.gt=blind.gt #三维

        # # 物理先验
        # self.abundance_lr = abundance_lr  # [B, n_end, h, w] from Lr-HSI
        # self.endmember_lr = endmember_lr  # [1, C, n_end]
        # self.abundance_hr = abundance_hr  # [B, n_end, H, W] from Hr-MSI
        # self.endmember_hr = endmember_hr  # [1, c, n_end]
        #
        psf_est = np.reshape(psf, newshape=(1, 1, self.args.scale_factor, self.args.scale_factor)) #1 1 ratio ratio 大小的tensor
        self.psf_est = torch.tensor(psf_est).to(self.args.device).float()
        srf_est = np.reshape(srf.T, newshape=(srf.shape[1], srf.shape[0], 1, 1)) #self.srf.T 有一个T转置 (8, 191, 1, 1)
        self.srf_est = torch.tensor(srf_est).to(self.args.device).float()             # ms_band hs_bands 1 1 的tensor torch.Size([8, 191, 1, 1])

        self.psf_down=PSF_down() #__call__(self, input_tensor, psf, ratio):
        self.srf_down=SRF_down() #__call__(self, input_tensor, srf):

        self.noise1 = self.get_noise(self.gt.shape[2],(self.gt.shape[0],self.gt.shape[1])).to(self.args.device).float()
        self.noise2 = self.get_noise(self.gt.shape[2],(self.gt.shape[0],self.gt.shape[1])).to(self.args.device).float()
        #
        # # 网络结构：复杂多分支物理门控融合
        # in_ch = self.lr_hsi_fused.shape[1]
        # n_end = self.abundance_lr.shape[1] if self.abundance_lr is not None else 4
        # self.net = MultiBranchNet(in_ch, n_endmember=n_end)
        self.net = DualBranchAsymNet(hrhsi_input1.shape[1], self.args).to(self.args.device)

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch +  1 - self.args.niter3_dip) / float(self.args.niter_decay3_dip + 1)
            return lr_l


        self.optimizer=optim.Adam(self.net.parameters(), lr=self.args.lr_stage3_dip)
        self.scheduler=lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)
        
    def get_noise(self,input_depth, spatial_size, method='2D',noise_type='u', var=1./10):
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
        flag_best_1=[10,0,'data',0] #第一个是SAM，第二个是PSNR,第三个为恢复的图像,第四个是保存最佳结果对应的epoch
        flag_best_2=[10,0,'data',0]

        L1Loss = nn.L1Loss(reduction='mean')


        for epoch in range(1, self.args.niter3_dip + self.args.niter_decay3_dip + 1):
        
            
            self.optimizer.zero_grad()  #清空梯度
            self.hrhsi_final, self.hrhsi_trans, self.hrhsi_unet = self.net(self.hrhsi_input1, self.hrhsi_input2)



            ''' generate hr_msi_est '''
            #print(self.hrhsi_est.shape)
            self.hr_msi_hrhsi_trans = self.srf_down(self.hrhsi_trans,self.srf_est)    # 光谱退化
            self.hr_msi_hrhsi_unet = self.srf_down(self.hrhsi_unet,self.srf_est)

            ''' generate lr_hsi_est '''
            self.lr_hsi_hrhsi_trans = self.psf_down(self.hrhsi_trans, self.psf_est, self.args.scale_factor)   # 空间退化
            self.lr_hsi_hrhsi_unet = self.psf_down(self.hrhsi_unet, self.psf_est, self.args.scale_factor)
            



            # Y--->lr_hsi，Z--->hr_msi
            # 融合图像空间退化--->lr_hsi_hrhsi_trans，融合图像光谱退化--->hr_msi_hrhsi_unet
            loss_fhsi= L1Loss(self.hr_msi,self.hr_msi_hrhsi_trans) + L1Loss(self.lr_hsi,self.lr_hsi_hrhsi_trans)      # 空间退化+光谱退化
            loss_fmsi= L1Loss(self.hr_msi,self.hr_msi_hrhsi_unet) + L1Loss(self.lr_hsi,self.lr_hsi_hrhsi_unet)
            loss=loss_fhsi+loss_fmsi

            # === NEW: 加入Tucker秩正则 ===
            # loss += tucker_rank_reg(self.net, lambda_rank=getattr(self.args, 'lambda_rank', 1e-5))

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if epoch % 50 ==0:
                with torch.no_grad():
                    print("___________________stage-3__________________________")
                    print('epoch:{} lr:{}'.format(epoch,self.optimizer.param_groups[0]['lr']))
                    print('************')
                    # self.print_tucker_ranks()
                    
                    #转为W H C的numpy 方便计算指标
                    #hrmsi
                    hr_msi_numpy=self.hr_msi.data.cpu().detach().numpy()[0].transpose(1,2,0)
                    hr_msi_est_ftrans_numpy=self.hr_msi_hrhsi_trans.data.cpu().detach().numpy()[0].transpose(1,2,0)
                    hr_msi_est_funet_numpy=self.hr_msi_hrhsi_unet.data.cpu().detach().numpy()[0].transpose(1,2,0)
                    
                    #lrhsi
                    lr_hsi_numpy=self.lr_hsi.data.cpu().detach().numpy()[0].transpose(1,2,0)
                    lr_hsi_est_ftrans_numpy=self.lr_hsi_hrhsi_trans.data.cpu().detach().numpy()[0].transpose(1,2,0)
                    lr_hsi_est_funet_numpy=self.lr_hsi_hrhsi_unet.data.cpu().detach().numpy()[0].transpose(1,2,0)

                    #gt
                    hrhsi_est_numpy_ftrans=self.hrhsi_trans.data.cpu().detach().numpy()[0].transpose(1,2,0)
                    hrhsi_est_numpy_funet=self.hrhsi_unet.data.cpu().detach().numpy()[0].transpose(1,2,0)

                    # final
                    hrhsi_est_numpy_final=self.hrhsi_final.data.cpu().detach().numpy()[0].transpose(1,2,0)
                    #self.gt


                
                    ''' for ftrans'''
                    
                    #学习到的lrhsi与真值
                    sam,psnr,ergas,cc,rmse,Ssim,Uqi=MetricsCal(lr_hsi_numpy,lr_hsi_est_ftrans_numpy, self.args.scale_factor)
                    L1=np.mean( np.abs( lr_hsi_numpy - lr_hsi_est_ftrans_numpy ))
                    information1="生成lrhsi_ftrans与目标lrhsi\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1,sam,psnr,ergas,cc,rmse,Ssim,Uqi)
                    print(information1) #监控训练过程
                    print('************')
                
                    #学习到的hrmsi与真值
                    sam,psnr,ergas,cc,rmse,Ssim,Uqi=MetricsCal(hr_msi_numpy,hr_msi_est_ftrans_numpy, self.args.scale_factor)
                    L1=np.mean( np.abs( hr_msi_numpy - hr_msi_est_ftrans_numpy ))
                    information2="生成hrmsi_ftrans与目标hrmsi\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1,sam,psnr,ergas,cc,rmse,Ssim,Uqi)
                    print(information2) #监控训练过程
                    print('************')
                    
                    
                    #学习到的gt与真值
                    sam,psnr,ergas,cc,rmse,Ssim,Uqi=MetricsCal(self.gt,hrhsi_est_numpy_ftrans, self.args.scale_factor)
                    L1=np.mean( np.abs( self.gt - hrhsi_est_numpy_ftrans ))
                    information3="生成hrhsi_est_ftrans与目标hrhsi\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1,sam,psnr,ergas,cc,rmse,Ssim,Uqi)
                    print(information3) #监控训练过程
                    print('************')
                   
                    file_name = os.path.join(self.args.expr_dir, 'Stage3.txt')
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
                  
                        flag_best_1[0]=sam
                        flag_best_1[1]=psnr
                        flag_best_1[2]=self.hrhsi_trans #保存四维tensor
                        flag_best_1[3]=epoch
                        
                        information_a=information1
                        information_b=information2
                        information_c=information3
                                               
                    ''' for ftrans'''
                    
                    print('--------------------------------')
                    
                    ''' for funet'''
                    #学习到的lrhsi与真值
                    sam,psnr,ergas,cc,rmse,Ssim,Uqi=MetricsCal(lr_hsi_numpy,lr_hsi_est_funet_numpy, self.args.scale_factor)
                    L1=np.mean( np.abs( lr_hsi_numpy - lr_hsi_est_funet_numpy ))
                    information1="生成lrhsi_funet与目标lrhsi\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1,sam,psnr,ergas,cc,rmse,Ssim,Uqi)
                    print(information1) #监控训练过程
                    print('************')
                
                    #学习到的hrmsi与真值
                    sam,psnr,ergas,cc,rmse,Ssim,Uqi=MetricsCal(hr_msi_numpy,hr_msi_est_funet_numpy, self.args.scale_factor)
                    L1=np.mean( np.abs( hr_msi_numpy - hr_msi_est_funet_numpy ))
                    information2="生成hrmsi_fmsi与目标hrmsi\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1,sam,psnr,ergas,cc,rmse,Ssim,Uqi)
                    print(information2) #监控训练过程
                    print('************')
                    
                    
                    #学习到的gt与真值
                    sam,psnr,ergas,cc,rmse,Ssim,Uqi=MetricsCal(self.gt,hrhsi_est_numpy_funet, self.args.scale_factor)
                    L1=np.mean( np.abs( self.gt - hrhsi_est_numpy_funet ))
                    information3="生成hrhsi_est_fmsi与目标hrhsi\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1,sam,psnr,ergas,cc,rmse,Ssim,Uqi)
                    print(information3) #监控训练过程
                    print('************')
                   
                    print('————————————————————————————————')
                    
                    file_name = os.path.join(self.args.expr_dir, 'Stage3.txt')
                    with open(file_name, 'a') as opt_file:
                        
                        #opt_file.write('epoch:{}'.format(epoch))
                        opt_file.write('-------------------------------')
                        opt_file.write('\n')
                        opt_file.write(information1)
                        opt_file.write('\n')
                        opt_file.write(information2)
                        opt_file.write('\n')
                        opt_file.write(information3)
                        opt_file.write('\n')
                        #opt_file.write('————————————————————————————————')
                    # print(type(flag_best_1[2]))
                    # print(flag_best_1[2])
                    if sam < flag_best_2[0] and psnr > flag_best_2[1]:
                    
                        flag_best_2[0]=sam
                        flag_best_2[1]=psnr
                        flag_best_2[2]=self.hrhsi_unet #保存四维tensor
                        flag_best_2[3]=epoch
                        
                        information_d=information1
                        information_e=information2
                        information_f=information3                   
                    ''' for fmsi'''



        #保存最好的结果
        # scipy.io.savemat(os.path.join(self.args.expr_dir, 'Out_fhsi_S3.mat'), {'Out':flag_best_1[2].data.cpu().numpy()[0].transpose(1,2,0)})
        # scipy.io.savemat(os.path.join(self.args.expr_dir, 'Out_fmsi_S3.mat'), {'Out':flag_best_2[2].data.cpu().numpy()[0].transpose(1,2,0)})

        if isinstance(flag_best_1[2], torch.Tensor):
            scipy.io.savemat(
                os.path.join(self.args.expr_dir, 'hrhsi_1_S3.mat'),
                {'Out': flag_best_1[2].data.cpu().numpy()[0].transpose(1, 2, 0)}
            )
        else:
            print("Warning: flag_best_1[2] is not a tensor, skipping save.")
        if isinstance(flag_best_2[2], torch.Tensor):
            scipy.io.savemat(
                os.path.join(self.args.expr_dir, 'hrhsi_2_S3.mat'),
                {'Out': flag_best_2[2].data.cpu().numpy()[0].transpose(1, 2, 0)}
            )
        else:
            print("Warning: flag_best_2[2] is not a tensor, skipping save.")
        #保存精度
        file_name = os.path.join(self.args.expr_dir, 'Stage3.txt')
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


        return flag_best_1[2],flag_best_2[2]
 
if __name__ == "__main__":
    
    pass
    
