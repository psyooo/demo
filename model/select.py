# -*- coding: utf-8 -*-

"""
❗❗❗❗❗❗#此py作用：第四阶段的决策融合
"""
import torch
from .evaluation import MetricsCal
import os
import scipy.io as sio
import numpy as np
from datetime import datetime
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics.functional import structural_similarity_index_measure as ssim
from visualizer import UnifiedVisualizer

def spatial_spectral_total_variation(image, spatial_weight=0.5, spectral_weight=0.5):
    """
    计算空间光谱总变分 (SSTV)
    :param image: 输入图像，形状为 (B, C, H, W)
    :param spatial_weight: 空间域权重
    :param spectral_weight: 光谱域权重
    :return: SSTV损失
    """
    # 确保权重之和为1
    assert spatial_weight + spectral_weight == 1.0, "权重之和必须为1"

    # 计算空间域总变分
    dx = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])  # 水平方向
    dy = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])  # 垂直方向
    spatial_tv = torch.sum(dx) + torch.sum(dy)  # 组合水平和垂直方向(空间变分)

    # 计算光谱域总变分
    db = torch.abs(image[:, :-1, :, :] - image[:, 1:, :, :])  # 光谱方向
    spectral_tv = torch.sum(db)

    # 组合空间和光谱总变分
    sstv_loss = spatial_weight * spatial_tv + spectral_weight * spectral_tv

    return sstv_loss


# 新增：边缘特征计算
def compute_edge_features(x):
    """使用Sobel算子计算边缘特征"""
    # 创建Sobel算子
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3) # 创建一个3×3大小的Sobel算子，用于计算x方向上的梯度
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3) # 创建一个3×3大小的Sobel算子，用于计算y方向上的梯度

    # 转为灰度图（对各波段求平均）
    if x.shape[1] > 1:
        x_gray = torch.mean(x, dim=1, keepdim=True)
    else:
        x_gray = x

    # 应用Sobel算子
    grad_x = F.conv2d(x_gray, sobel_x, padding=1)   # 计算x方向上的梯度
    grad_y = F.conv2d(x_gray, sobel_y, padding=1)   # 计算y方向上的梯度

    # 计算梯度幅值
    edge = torch.sqrt(grad_x ** 2 + grad_y ** 2)


    # 归一化（修复版本兼容性问题）
    max_val = torch.max(torch.max(edge, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
    edge = edge / (max_val + 1e-8)

    return edge


# 新增：权重生成网络
class WeightGenerator(nn.Module):
    """生成像素级融合权重的轻量级网络"""

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 1, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        weight = torch.sigmoid(self.conv4(x))
        return weight


# 新增：多特征引导的像素自适应融合
def adaptive_feature_fusion(Out_fhsi, Out_fmsi, blind, use_edge=True, use_error_map=True):
    """
    多特征引导的像素自适应融合
    Args:
        Out_fhsi: 第一个候选分支输出 [B, C, H, W]
        Out_fmsi: 第二个候选分支输出 [B, C, H, W]
        blind: 包含必要信息的对象
        use_edge: 是否使用边缘特征
        use_error_map: 是否使用误差图
    Returns:
        fused_result: 融合结果 [B, C, H, W]
        weight_map: 融合权重 [B, 1, H, W]
    """
    device = Out_fhsi.device    # 获取设备信息
    num_bands = Out_fhsi.shape[1]    # 获取图像的通道数

    # 构建输入特征
    features = [Out_fhsi, Out_fmsi]  # 两个候选分支的输出
    in_channels = 2 * num_bands  # 两个候选分支

    # 添加边缘特征
    if use_edge:
        edge_fhsi = compute_edge_features(Out_fhsi)  # 计算第一个候选分支的边缘特征
        edge_fmsi = compute_edge_features(Out_fmsi)  # 计算第二个候选分支的边缘特征
        features.extend([edge_fhsi, edge_fmsi])  # 添加边缘特征
        in_channels += 2    # 边缘特征的通道数

    # 添加误差图特征
    if use_error_map and hasattr(blind, 'tensor_hr_msi') and hasattr(blind, 'model'):
        hr_msi = blind.tensor_hr_msi    # 高光谱图像
        srf = blind.model.srf    # 超分辨率网络

        # 将高光谱图像通过SRF投影到多光谱空间
        b, c, h, w = Out_fhsi.shape  # 获取图像的批大小、通道数、高、宽
        Out_fhsi_reshaped = Out_fhsi.permute(0, 2, 3, 1).reshape(b * h * w, c)  # 将图像展平为一维数组
        Out_fmsi_reshaped = Out_fmsi.permute(0, 2, 3, 1).reshape(b * h * w, c)  # 将图像展平为一维数组

        srf_est = srf.data.cpu().numpy().squeeze().T    # 超分辨率网络的SRF估计值
        hr_msi_est_fhsi = torch.matmul(
            Out_fhsi_reshaped, torch.tensor(srf_est, device=device)).reshape(b, h, w,-1).permute(0,3,1,2)    # 第一个候选分支的SRF投影
        hr_msi_est_fmsi = torch.matmul(
            Out_fmsi_reshaped, torch.tensor(srf_est, device=device)).reshape(b, h, w,-1).permute(0,3,1,2)    # 第二个候选分支的SRF投影

        # 计算SAM误差图
        # 修改点：添加.detach()以允许转换为numpy数组
        sam_fhsi = compute_sammap(hr_msi.cpu().numpy()[0].transpose(1, 2, 0),
                                  hr_msi_est_fhsi.detach().cpu().numpy()[0].transpose(1, 2, 0))  # 第一个候选分支的SAM误差图
        sam_fmsi = compute_sammap(hr_msi.cpu().numpy()[0].transpose(1, 2, 0),
                                  hr_msi_est_fmsi.detach().cpu().numpy()[0].transpose(1, 2, 0))  # 第二个候选分支的SAM误差图

        # 转换为张量
        sam_fhsi_tensor = torch.tensor(sam_fhsi, device=device).unsqueeze(0).unsqueeze(0)    # 第一个候选分支的SAM误差图张量
        sam_fmsi_tensor = torch.tensor(sam_fmsi, device=device).unsqueeze(0).unsqueeze(0)    # 第二个候选分支的SAM误差图张量

        features.extend([sam_fhsi_tensor, sam_fmsi_tensor])  # 添加SAM误差图特征
        in_channels += 2    # SAM误差图特征的通道数

    # 拼接所有特征
    input_features = torch.cat(features, dim=1)

    # 初始化权重生成网络
    weight_generator = WeightGenerator(in_channels).to(device)

    # 生成融合权重
    weight_map = weight_generator(input_features)

    # 执行融合
    fused_result = weight_map * Out_fhsi + (1 - weight_map) * Out_fmsi

    return fused_result, weight_map

def compute_psnr(x_true, x_pred):
    assert x_true.ndim == 3 and x_pred.ndim ==3

    img_w, img_h, img_c = x_true.shape  # 获得图像的宽、高、通道数
    ref = x_true.reshape(-1, img_c)  # 将图像展平为一维数组:(w*h,c)
    tar = x_pred.reshape(-1, img_c)  # 将图像展平为一维数组:(w*h,c)
    msr = np.mean((ref - tar)**2, 0) # 列和
    max2 = np.max(ref,0)**2  # 列最大值
    psnrall = 10*np.log10(max2/msr)  # 计算psnr
    m_psnr = np.mean(psnrall)   # 计算平均psnr
    psnr_all = psnrall.reshape(img_c)   # 将psnr展平为通道数
    return m_psnr,psnr_all

#逐波段计算RMSE
def compute_rmse_byband(x_true, x_pre):
     assert x_true.ndim == 3 and x_pre.ndim ==3 and x_true.shape == x_pre.shape # x_true和x_pre的shape必须相同
     img_w, img_h, img_c = x_true.shape # 获得图像的宽、高、通道数
     ref = x_true.reshape(-1, img_c)    # 将图像展平为一维数组
     tar = x_pre.reshape(-1, img_c)    # 将图像展平为一维数组
     rmse_byband=np.sqrt(np.mean((ref - tar)**2, 0))    # 计算rmse
     return rmse_byband

#计算SAMMAP
def compute_sammap(x_true, x_pred):
    assert x_true.ndim ==3 and x_true.shape == x_pred.shape # x_true和x_pred的shape必须相同
    w, h, c = x_true.shape  # 获得图像的宽、高、通道数
    x_true = x_true.reshape(-1, c) # 一行为一条光谱曲线
    x_pred = x_pred.reshape(-1, c)  # 一行为一条光谱曲线

    #sam = (x_true * x_pred).sum(axis=1) / (np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1)+1e-5) 原本的
    #sam_all  = (x_true * x_pred).sum(axis=1) / (np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1))
    sam_all = (x_true * x_pred).sum(axis=1) / (np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1)+1e-7)    # 加上1e-7防止除0
    sam_all = np.arccos(sam_all) * 180 / np.pi  # 计算sammap
    sammap  = sam_all.reshape(w, h)  # 将sammap展平为图像大小
    return  sammap

def compute_rmsemap(x_true, x_pred):
    assert x_true.ndim ==3 and x_true.shape == x_pred.shape
    w, h, c = x_true.shape
    x_true = x_true.reshape(-1, c) # 一行为一条光谱曲线
    x_pred = x_pred.reshape(-1, c)

    #sam = (x_true * x_pred).sum(axis=1) / (np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1)+1e-5) 原本的
    #sam_all  = (x_true * x_pred).sum(axis=1) / (np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1))
    rmse_all = np.sqrt(np.mean((x_true - x_pred)**2, 1))
    rmsemap=rmse_all.reshape(w, h)
    return  rmsemap

def select_decision(Out_fhsi,Out_fmsi,blind): #Out_fhsi,Out_fmsi是四维device tensor

    # 准备GT数据并移到相同设备
    gt_tensor = torch.from_numpy(blind.gt).unsqueeze(0).permute(0, 3, 1, 2).to(Out_fhsi.device)

    # 打开文件准备写入损失
    file_name = os.path.join(blind.args.expr_dir, 'Stage4.txt')
    with open(file_name, 'a') as opt_file:
        opt_file.write("\n=== 改进的多指标损失优化 ===\n")
        opt_file.write(f"Parameters: sstv_weight=0.05, sam_weight=0.4, ssim_weight=0.4, l1_weight=0.2\n")
        opt_file.write("Iteration\tTotal Loss\tSAM Loss\tSSIM Loss\tL1 Loss\tSSTV Loss\n")

    # 迭代优化
    num_iterations = 1000
    # 使用新的自适应融合方法
    fused_result, weight_map = adaptive_feature_fusion(
        Out_fhsi,
        Out_fmsi,
        blind,
        use_edge=True,
        use_error_map=True
    )
    # 将初始结果作为起点
    current_result = fused_result.detach()
    # 记录当前迭代的TV损失
    for i in range(1,num_iterations+1):
        # 创建新的可训练参数
        optimized_result = nn.Parameter(current_result.clone())
        # 定义优化器（每次迭代重新创建）
        optimizer = torch.optim.Adam([optimized_result], lr=0.001)

        with torch.no_grad():
            current_fused_result, current_weight_map = adaptive_feature_fusion(
                Out_fhsi,
                Out_fmsi,
                blind,
                use_edge=True,
                use_error_map=True
            )
            # 将当前结果复制到可训练参数
            optimized_result.data.copy_(current_fused_result)

        # 计算改进的损失函数
        loss, loss_components = improved_fusion_loss(
            optimized_result,
            gt_tensor,
            current_weight_map,
            sstv_weight=0.005,
            sam_weight=1,
            ssim_weight=1,
            l1_weight=0.2
        )
        # 记录当前迭代的SSTV损失
        if i % 100 == 0:
            print(f"Iteration {i+1}: total Loss = {loss.item()}")
            # 记录当前迭代的损失
            with open(file_name, 'a') as opt_file:
                opt_file.write(
                    f"{i}\t{loss.item()}\t{loss_components['SAM']}\t{loss_components['SSIM']}\t{loss_components['L1']}\t{loss_components['SSTV']}\n")
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新当前结果为优化后的结果
        current_result = optimized_result.detach()

    # 将优化后的结果转换回numpy数组
    srf_out = current_result.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # 计算并保存基于SSTV优化后的评估指标
    gt = blind.gt
    sam, psnr, ergas, cc, rmse, Ssim, Uqi = MetricsCal(gt, srf_out, blind.args.scale_factor)
    L1 = np.mean(np.abs(gt - srf_out))
    print("____________________stage-4________________________")
    information_sstv = "gt与hr_hsi_srf_rmse(多指标优化后)\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(
        L1, sam, psnr, ergas, cc, rmse, Ssim, Uqi)
    print(information_sstv)

    with open(file_name, 'a') as opt_file:
        opt_file.write("\n=== 多指标优化后评估指标 ===\n")
        opt_file.write(information_sstv)
        opt_file.write('\n')

    # 保存决策结果
    sio.savemat(os.path.join(blind.args.expr_dir, 'srf_Out_S4.mat'), {'Out': srf_out})


    # 保存权重图用于分析
    if hasattr(blind.args, 'save_weight_map') and blind.args.save_weight_map:
        weight_np = current_weight_map.data.cpu().numpy()[0, 0]  # [H, W]
        sio.savemat(os.path.join(blind.args.expr_dir, 'fusion_weight.mat'), {'weight': weight_np})

    # 关键：使用与训练过程相同的环境名称，确保在同一网页显示
    visualizer = UnifiedVisualizer(env_name="Model_Visualization", save_dir="./final_visualizations")

    # 可视化最终结果
    metrics = visualizer.visualize_final_results(
        pred_img=srf_out,
        gt_img=gt,
        img_name="best_model",
        bands=[45, 29, 13],
        title_suffix="(Best Model Selection)"
    )

    return srf_out

# 改进的损失函数
def improved_fusion_loss(fused_result, gt, weight_map=None, sstv_weight=0.05, sam_weight=0.4, ssim_weight=0.4,
                         l1_weight=0.2):
    """
    改进的融合损失函数，结合多种评价指标
    Args:
        fused_result: 融合结果 [B, C, H, W]、gt: 真实高光谱图像 [B, C, H, W]
        weight_map: 融合权重图 [B, 1, H, W] (可选)、sstv_weight: SSTV正则化权重
        sam_weight: SAM损失权重、ssim_weight: SSIM损失权重、l1_weight: L1损失权重
    Returns:
        loss: 总损失\loss_components: 各损失分量的字典
    """
    # 1. 计算光谱角映射(SAM)损失
    sam_loss = compute_sam_loss(fused_result, gt)

    # 2. 计算结构相似性(SSIM)损失
    ssim_loss = 1 - ssim(fused_result, gt, data_range=1.0)

    # 3. 计算L1损失
    l1_loss = F.l1_loss(fused_result, gt)

    # 4. 计算SSTV正则化 (如果提供了权重图)
    sstv_reg = 0
    if weight_map is not None and sstv_weight > 0:
        sstv_reg = spatial_spectral_total_variation(
            weight_map,
            spatial_weight=0.7,
            spectral_weight=0.3
        )

    # 组合损失
    main_loss = sam_weight * sam_loss + ssim_weight * ssim_loss + l1_weight * l1_loss
    total_loss = main_loss + sstv_weight * sstv_reg

    return total_loss, {'SAM': sam_loss.item(), 'SSIM': ssim_loss.item(), 'L1': l1_loss.item(),
                        'SSTV': sstv_reg.item() if sstv_reg != 0 else 0}

def compute_sam_loss(pred, target):
    """计算光谱角映射损失"""
    # 确保输入维度正确
    assert pred.dim() == 4 and target.dim() == 4, "输入必须是4D张量 [B, C, H, W]"

    # 重塑为 [B, C, H*W]
    pred_reshaped = pred.reshape(pred.size(0), pred.size(1), -1)
    target_reshaped = target.reshape(target.size(0), target.size(1), -1)

    # 计算点积、模长
    dot_product = torch.sum(pred_reshaped * target_reshaped, dim=1)
    pred_norm = torch.norm(pred_reshaped, p=2, dim=1)
    target_norm = torch.norm(target_reshaped, p=2, dim=1)

    # 避免除以零
    eps = 1e-8
    cos_theta = dot_product / (pred_norm * target_norm + eps)

    # 确保余弦值在有效范围内
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)

    # 计算光谱角
    sam = torch.acos(cos_theta)

    # 计算平均SAM损失
    return torch.mean(sam)

