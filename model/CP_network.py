import torch
import torch.nn as nn
import torch.nn.functional as F
from .network_s3_1 import DilatedConvBlock, SimpleUNet


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=False))

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)



class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        self.avgpool_conv = nn.Sequential(nn.AvgPool2d(2), DoubleConv(in_channels, out_channels, use_bn))

    def forward(self, x):
        return self.avgpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, use_bn=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, use_bn)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, use_bn)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is not None:
            diffY = x2.size()[2] - x1.size()[2]  # 计算x2和x1在Y轴上的差值
            diffX = x2.size()[3] - x1.size()[3]  # 计算x2和x1在X轴上的差值
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])    # 填充x1使其和x2的尺寸相同
            x = torch.cat([x2, x1], dim=1)   # 拼接x1和x2
        else:
            x = x1
        return self.conv(x)


class OutConv(nn.Module):
    """输出卷积层，调整通道数为光谱波段数"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.Sigmoid(self.conv(x))


class FRM(nn.Module):
    """特征细化模块（FRM）"""

    def __init__(self, in_channels, out_channels):
        super(FRM, self).__init__()
        self.out_channels = out_channels
        # 修改第一个卷积层的输入通道数为in_channels*2
        self.conv1 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(inplace=False)

    def forward(self, x, skip_x, map):
        # x, skip_x: [B, C, H, W]，空间可能不一致
        if x.shape[2:] != skip_x.shape[2:]:
            x = F.interpolate(x, size=skip_x.shape[2:], mode='bilinear', align_corners=True)
        concat_feat = torch.cat([x, skip_x], dim=1)

        out = self.leaky_relu(self.conv1(concat_feat) * map)
        out = self.leaky_relu(self.conv2(out)+x)
        return out


class MDRMWithCPRecon(nn.Module):
    """改进版 MDRM with CP Reconstruction"""
    def __init__(self, in_channels, k=4):
        super(MDRMWithCPRecon, self).__init__()
        self.k = k
        self.in_channels = in_channels

        # 初始融合
        self.conv3x3 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)

        # 调整因子维度适配器（用于 avg+max 池化结果）
        self.adapter_feat1 = nn.Conv2d(2, in_channels, kernel_size=1)
        self.adapter_feat2 = nn.Conv2d(2, in_channels, kernel_size=1)
        self.adapter_feat3 = nn.Conv2d(2, in_channels, kernel_size=1)

        # 注意力生成器
        self.U_gen = nn.Conv2d(in_channels, k, kernel_size=1)
        self.u1_conv = nn.Conv2d(in_channels, in_channels*k, kernel_size=1)
        self.u2_conv = nn.Conv2d(in_channels, in_channels*k, kernel_size=1)
        self.u3_conv = nn.Conv2d(in_channels, in_channels*k, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

        # 重建映射
        self.recon_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 空间注意力
        self.spatial_conv = nn.Conv2d(1, 1, kernel_size=1)

        # 光谱注意力
        self.spectral_conv_avg = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.spectral_conv_max = nn.Conv1d(in_channels, in_channels, kernel_size=1)

        # 可学习融合权重
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始值为0.5
        self.lambda_weight = nn.Parameter(torch.ones(k))  # [k]

        self.leaky_relu = nn.LeakyReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()

    def _compute_group_sparsity(self, U1, U2, U3):
        """
        U1: [B, C, k], U2: [B, H, k], U3: [B, W, k]
        返回 L2,1 组稀疏正则：
        sum_r sqrt( ||U1[:,:,r]||_F^2 + ||U2[:,:,r]||_F^2 + ||U3[:,:,r]||_F^2 )
        """
        # 对 batch 维与特征维做 Frobenius，再对秩维聚合
        U1_f = torch.linalg.norm(U1, ord='fro', dim=(0, 1))  # [k]
        U2_f = torch.linalg.norm(U2, ord='fro', dim=(0, 1))  # [k]
        U3_f = torch.linalg.norm(U3, ord='fro', dim=(0, 1))  # [k]
        group = torch.sqrt(U1_f ** 2 + U2_f ** 2 + U3_f ** 2 + 1e-12)  # [k]
        return group.sum()

    def forward(self, frm_feat, other_feat):
        batch, C, H, W = frm_feat.shape

        concat_feat = torch.cat([frm_feat, other_feat], dim=1)
        Fm = self.leaky_relu(self.conv3x3(concat_feat))  # [B, C, H, W]

        # print("Fm shape:", Fm.shape)  # 你融合特征的shape
        # print("u1_conv out shape:", self.u1_conv(Fm).shape)
        # print("reshape目标：", (batch, C, self.k, H, W))

        # ===== 模式1: 通道维度 =====
        mode1 = Fm.permute(0, 2, 3, 1).reshape(batch, H * W, C).permute(0, 2, 1)  # [B, C, H*W]
        avg1 = torch.mean(mode1, dim=2, keepdim=True)
        max1 = torch.max(mode1, dim=2, keepdim=True)[0]
        feat1 = torch.cat([avg1, max1], dim=2).permute(0, 2, 1).unsqueeze(-1)  # [B, 2, C, 1]
        feat1 = self.adapter_feat1(feat1)
        U1 = self.softmax(self.U_gen(feat1).squeeze(-1).permute(0, 2, 1))  # [B, C, k]

        # ===== 模式2: 宽度维度 =====
        mode2 = Fm.permute(0, 3, 1, 2).reshape(batch, W, C * H)
        avg2 = torch.mean(mode2, dim=2, keepdim=True)
        max2 = torch.max(mode2, dim=2, keepdim=True)[0]
        feat2 = torch.cat([avg2, max2], dim=2).permute(0, 2, 1).unsqueeze(-1)  # [B, 2, W, 1]
        feat2 = self.adapter_feat2(feat2)
        U2 = self.softmax(self.U_gen(feat2).squeeze(-1).permute(0, 2, 1))  # [B, W, k]

        # ===== 模式3: 高度维度 =====
        mode3 = Fm.permute(0, 2, 1, 3).reshape(batch, H, C * W)
        avg3 = torch.mean(mode3, dim=2, keepdim=True)
        max3 = torch.max(mode3, dim=2, keepdim=True)[0]
        feat3 = torch.cat([avg3, max3], dim=2).permute(0, 2, 1).unsqueeze(-1)  # [B, 2, H, 1]
        feat3 = self.adapter_feat3(feat3)
        U3 = self.softmax(self.U_gen(feat3).squeeze(-1).permute(0, 2, 1))  # [B, H, k]

        # ===== 空间注意力 =====
        spatial_att = self.sigmoid(self.spatial_conv(torch.matmul(U3, U2.transpose(1, 2)).unsqueeze(1)))  # [B, 1, H, W]

        # ===== 光谱注意力 =====
        # 1. U2, U3拼接
        U2U3_cat = torch.cat([U2, U3], dim=1)  # [B, W+H, k]

        # 2. U1 × U2U3_cat^T  (注意k维一致)
        F_spe = torch.bmm(U1, U2U3_cat.transpose(1, 2))  # [B, C, W+H]

        # 全局平均池化和最大池化
        glb_avg = torch.mean(F_spe, dim=-1, keepdim=True)  # [B, C, 1, 1]
        glb_max = torch.amax(F_spe, dim=-1, keepdim=True)  # [B, C, 1, 1]

        glb_avg_out = self.spectral_conv_avg(glb_avg)  # [B, C, 1]
        glb_max_out = self.spectral_conv_max(glb_max)  # [B, C, 1]

        # 对两个全局特征执行1×1卷积
        spectral_att = self.sigmoid(glb_avg_out + glb_max_out)  # [B, C, 1]
        spectral_att = spectral_att.unsqueeze(-1)  # [B, C, 1, 1]

        spectral_att = self.sigmoid(spectral_att) # [B, 1, C]

        Weight = spectral_att * spatial_att  # [B, 1, H, W]

        fused_feat = self.alpha * Weight * frm_feat + (1 - self.alpha) * (1 - Weight) * other_feat



        # ===== 重建映射 =====
        cp_recon = torch.einsum('bcr,bhr,bwr,r->bchw', U1, U2, U3, self.lambda_weight)
        cp_recon = (self.recon_conv(cp_recon)) * Weight + Fm  # [B, C, H, W]

        # self.reg_l21, energy_vec= self._compute_group_sparsity(U1, U2, U3)  # L2,1 组稀疏
        # self.reg_l1_lambda = torch.norm(self.lambda_weight, p=1)  # λ 的 L1 稀疏
        # self.rank_energy = energy_vec.detach()  # [k] 供外部剪枝参考

        return fused_feat, cp_recon

class DenseFusionBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_rate, 3, padding=1)
        self.se1 = CBAMBlock(growth_rate)
        self.conv2 = nn.Conv2d(in_channels + growth_rate, growth_rate, 3, padding=1)
        self.se2 = CBAMBlock(growth_rate)
        self.conv3 = nn.Conv2d(in_channels + 2 * growth_rate, out_channels, 3, padding=1)
        self.se3 = CBAMBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, *inputs):
        # inputs: [f1, f2, ...]
        x = torch.cat(inputs, dim=1)
        y1 = self.se1(self.relu(self.conv1(x)))
        y2 = self.se2(self.relu(self.conv2(torch.cat([x, y1], dim=1))))
        y3 = self.se3(self.relu(self.conv3(torch.cat([x, y1, y2], dim=1))))
        return y3

class CBAMBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(ch, ch // r, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(ch // r, ch, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        # 空间注意力
        self.conv_spatial = nn.Conv2d(2, 1, 7, padding=3)
    def forward(self, x):
        # 通道注意力
        avg = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_ = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        ca = self.sigmoid(avg + max_)
        x = x * ca
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.sigmoid(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * sa
        return x

class spa_attn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv2 = nn.Conv2d(in_channels * 2, out_channels, 1)
        self.conv3 = nn.Conv2d(out_channels, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        cat_out = torch.cat([avg_out, max_out], dim=1)
        cat_out = self.conv2(cat_out)
        cat_out = self.relu(cat_out).expand(-1, -1, x.size(2), x.size(3))
        # cat_out = self.norm(cat_out)
        attn_out = self.sigmoid(self.conv3(cat_out))
        # x = x * attn_out

        return attn_out

class spe_attn(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # 通道压缩比
        mid_channels = max(in_channels // reduction, 4)
        # 分支1：AvgPool
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels, bias=False)
        )
        # 分支2：MaxPool
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc2 = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()
        # AvgPool分支
        avg = self.avg_pool(x).view(B, C)
        avg_out = self.fc1(avg)
        # MaxPool分支
        maxp = self.max_pool(x).view(B, C)
        max_out = self.fc2(maxp)
        # 相加融合
        out = avg_out + max_out
        scale = self.sigmoid(out).view(B, C, 1, 1)
        return scale







class HSI_MSI_Fusion_UNet(nn.Module):
    """高光谱与多光谱图像融合U-Net网络（Hr-MSI下采样到Lr-HSI尺度）"""

    def __init__(self,args, hr_msi_channels, lr_hsi_channels, out_channels, bilinear=True, k=8):
        super().__init__()
        self.hr_msi_channels = hr_msi_channels
        self.lr_hsi_channels = lr_hsi_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.ReLU = nn.ReLU(inplace=False)
        factor = 2 if bilinear else 1

        # Hr-MSI分支的编码器（下采样到Lr-HSI尺度）
        self.hr_inc = DoubleConv(hr_msi_channels, 64)
        self.hr_down1 = Down(64, 128)
        self.hr_down2 = Down(128, 256)
        self.hr_down3 = Down(256, 512)
        # 最后一个下采样层禁用BN，防止尺寸过小导致错误
        self.hr_down4 = Down(512, 1024 // factor, use_bn=False)

        # Lr-HSI分支（调整通道数匹配Hr-MSI的hr_x4）
        self.lr_inc = DoubleConv(lr_hsi_channels, 512, use_bn=False)

        # 特征融合模块（在hr_x4和Lr-HSI尺度）
        self.frm1 = FRM(512,512)    # hr_x4 + lr_x, 拼接后 1024，输出 512
        self.frm2 = FRM(256,256)    # hr_x3 + skip, 拼接后 512，输出 256
        self.frm3 = FRM(128,128)
        self.frm4 = FRM(64,64)
        self.mdrm1 = MDRMWithCPRecon(512, k)
        self.mdrm2 = MDRMWithCPRecon(256, k)
        self.mdrm3 = MDRMWithCPRecon(128, k)
        self.mdrm4 = MDRMWithCPRecon(64, k)

        # 解码器
        self.up1 = Up(512 + 256, 256, bilinear)
        self.up2 = Up(256 + 128, 128, bilinear)
        self.up3 = Up(128 + 64, 64, bilinear)
        self.up4 = Up(64, 64, bilinear)

        # 特征融合层
        self.conv1 = nn.Conv2d(512, 64, 1)
        self.conv2 = nn.Conv2d(256, 64, 1)
        self.conv3 = nn.Conv2d(128, 64, 1)
        self.fuse3 = nn.Sequential(nn.Conv2d(64 * 2, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.fuse2 = nn.Sequential(nn.Conv2d(64 * 2, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.fuse1 = nn.Sequential(nn.Conv2d(64 * 2, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.out = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True))

        # 输出层，将CP重构特征和融合特征映射到Hr-HSI的维度
        self.cp_recon_mapping = nn.Sequential(DoubleConv(64, 64), OutConv(64, out_channels))

        self.fused_feat_mapping = nn.Sequential(DoubleConv(64, 64), OutConv(64, out_channels))
        self.Sigmoid = nn.Sigmoid()

        # 将所有特征都压到统一通道数
        self.reduce1 = nn.Conv2d(512, 64, 1)
        self.reduce2 = nn.Conv2d(256, 64, 1)
        self.reduce3 = nn.Conv2d(128, 64, 1)
        self.reduce4 = nn.Conv2d(64, 64, 1)
        # Dense融合块
        self.fusion = DenseFusionBlock(64 * 4, 64, 64)

        # 空洞卷积模块
        self.DCB1 = DilatedConvBlock(64, 3, [1, 2, 4])
        self.DCB2 = DilatedConvBlock(64, 3, [1, 2, 4])
        # 简单门控注意力U-Net
        self.SCAUNet = SimpleUNet(64, 64)

        self.spa_attn1 = spa_attn(512, 512)
        self.spa_attn2 = spa_attn(256, 256)
        self.spa_attn3 = spa_attn(128, 128)
        self.spa_attn4 = spa_attn(64, 64)

        self.spe_attn1 = spe_attn(512)
        self.spe_attn2 = spe_attn(256)
        self.spe_attn3 = spe_attn(128)
        self.spe_attn4 = spe_attn(64)





    def forward(self, hr_msi, lr_hsi):
        # Hr-MSI分支的编码路径
        hr_x1 = self.ReLU(self.hr_inc(hr_msi))  # (1, 64, 240, 240)
        hr_x2 = self.hr_down1(hr_x1)  # (1, 128, 120, 120)
        hr_x3 = self.hr_down2(hr_x2)  # (1, 256, 60, 60)
        hr_x4 = self.hr_down3(hr_x3)  # (1, 512, 30, 30)

        # 在hrmsi分支的每个下采样层后，应用空间注意力模块
        hr_x4_att_map = self.spa_attn1(hr_x4)
        hr_x3_att_map = self.spa_attn2(hr_x3)
        hr_x2_att_map = self.spa_attn3(hr_x2)
        hr_x1_att_map = self.spa_attn4(hr_x1)

        # Lr-HSI分支（调整通道数匹配hr_x4）
        lr_x = self.lr_inc(lr_hsi)  # (1, 512, 30, 30)
        lr_x_map = self.spe_attn1(lr_x)
        map_1 = self.Sigmoid(hr_x4_att_map * lr_x_map)
        map_1 = F.interpolate(map_1, size=lr_x.shape[2:], mode='bilinear', align_corners=True)


        # 在hr_x4和Lr-HSI尺度应用FRM和MDRM模块
        frm_out_1 = self.frm1(hr_x4, lr_x, map_1)         # (1, 512, 30, 30)
        fused_feat_1, cp_recon_1 = self.mdrm1(frm_out_1, lr_x)     # (1, 512, 30, 30)


        # 解码器路径
        x1 = self.up1(fused_feat_1, hr_x3)  # (1, 256, 60, 60)
        x1_map = self.spe_attn2(x1)
        map_2 = self.Sigmoid(hr_x3_att_map * x1_map)
        map_2 = F.interpolate(map_2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        # 在hr_x3和x1应用FRM和MDRM模块
        frm_out_2 = self.frm2(hr_x3, x1, map_2)  # (1, 256, 60, 60)
        fused_feat_2, cp_recon_2 = self.mdrm2(frm_out_2, x1)     # (1, 256, 60, 60)


        x2 = self.up2(fused_feat_2, hr_x2)   # (1, 128, 120, 120)
        x2_map = self.spe_attn3(x2)
        map_3 = self.Sigmoid(hr_x2_att_map * x2_map)
        map_3 = F.interpolate(map_3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        # 在hr_x2和x2应用FRM和MDRM模块
        frm_out_3 = self.frm3(hr_x2, x2, map_3)       # (1, 128, 120, 120)
        fused_feat_3, cp_recon_3 = self.mdrm3(frm_out_3, x2)     # (1, 128, 120, 120)

        x3 = self.up3(fused_feat_3, hr_x1)   # (1, 64, 240, 240)
        x3_map = self.spe_attn4(x3)
        map_4 = self.Sigmoid(hr_x1_att_map * x3_map)
        map_4 = F.interpolate(map_4, size=x3.shape[2:], mode='bilinear', align_corners=True)

        # 在hr_x1和x3应用FRM和MDRM模块
        frm_out_4 = self.frm4(hr_x1, x3, map_4)       # (1, 64, 240, 240)
        fused_feat_4, cp_recon_4 = self.mdrm4(frm_out_4, x3)     # (1, 64, 240, 240)

        x = self.out(fused_feat_4)  # (1, 64, 240, 240)

        """方案一："""
        y1 = self.conv1(cp_recon_1)      # cp_recon_1(1,512,30,30)---->y1(1, 64, 30, 30)
        y2 = self.conv2(cp_recon_2)      # cp_recon_2(1,256,60,60)---->y2(1, 64, 60, 60)
        y3 = self.conv3(cp_recon_3)      # cp_recon_3(1,128,120,120)----->y3(1, 64, 120, 120)
        y2_up = F.interpolate(y1, size=y2.shape[2:], mode='bilinear', align_corners=True)        # (1, 64, 60, 60)
        y2_fuse = self.fuse3(torch.cat([y2, y2_up], dim=1)) + y2    # (1, 64, 60, 60)

        y3_up = F.interpolate(y2_fuse, size=y3.shape[2:], mode='bilinear', align_corners=True)      # (1, 64, 120, 120)
        y3_fuse = self.fuse2(torch.cat([y3, y3_up], dim=1)) + y3    # (1, 64, 120, 120)

        y4_up = F.interpolate(y3_fuse, size=cp_recon_4.shape[2:], mode='bilinear', align_corners=True)   # (1, 64, 240, 240)
        y4_fuse = self.fuse1(torch.cat([cp_recon_4, y4_up], dim=1)) + cp_recon_4      # (1, 64, 240, 240)
        """方案一结束"""

        y = self.out(y4_fuse)  # (1, 64, 240, 240)
        x = self.DCB1(x)
        # y = self.SCAUNet(y)
        # x = self.SCAUNet(x)       # SCAUNet网络不行，要换
        y = self.DCB2(y)

        # 将CP重构特征和融合特征映射到Hr-HSI的维度
        mapped_cp_recon = self.cp_recon_mapping(y)    # (1,C,H,W)  (1,46,240,240)
        mapped_fused_feat = self.fused_feat_mapping(x)       # (1,C,H,W)

        return mapped_cp_recon, mapped_fused_feat


