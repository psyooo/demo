import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_bn)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


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

    def forward(self, x):
        return self.conv(x)


class FRM(nn.Module):
    """特征细化模块（FRM）"""

    def __init__(self, in_channels, out_channels):
        super(FRM, self).__init__()
        self.out_channels = out_channels
        # 修改第一个卷积层的输入通道数为in_channels*2
        self.conv1 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(inplace=False)

    def forward(self, x, skip_x):
        concat_feat = torch.cat([x, skip_x], dim=1)
        out = self.leaky_relu(self.conv1(concat_feat))
        out = self.leaky_relu(self.conv2(out))
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

        self.leaky_relu = nn.LeakyReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, frm_feat, other_feat):
        batch, C, H, W = frm_feat.shape
        concat_feat = torch.cat([frm_feat, other_feat], dim=1)
        Fm = self.leaky_relu(self.conv3x3(concat_feat))  # [B, C, H, W]

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

        # ===== CP重构 =====
        U1_exp = U1.permute(0, 2, 1).unsqueeze(3).unsqueeze(4)  # [B, k, C, 1, 1]
        U2_exp = U2.permute(0, 2, 1).unsqueeze(2).unsqueeze(4)  # [B, k, 1, W, 1]
        U3_exp = U3.permute(0, 2, 1).unsqueeze(2).unsqueeze(3)  # [B, k, 1, 1, H]

        cp_recon = (U1_exp * U2_exp * U3_exp).sum(dim=1)  # [B, C, W, H]
        cp_recon = cp_recon.permute(0, 1, 3, 2)  # → [B, C, H, W]
        cp_recon = self.recon_conv(cp_recon)

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

        W = spectral_att * spatial_att  # [B, 1, H, W]

        fused_feat = self.alpha * W * frm_feat + (1 - self.alpha) * (1 - W) * other_feat

        return fused_feat, cp_recon

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
        self.fuse3 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 输出层，将CP重构特征和融合特征映射到Hr-HSI的维度
        self.cp_recon_mapping = nn.Sequential(
            DoubleConv(64, 64),
            OutConv(64, out_channels)
        )

        self.fused_feat_mapping = nn.Sequential(
            DoubleConv(64, 64),
            OutConv(64, out_channels)

        )
        self.Sigmoid = nn.Sigmoid()

    def forward(self, hr_msi, lr_hsi):
        # Hr-MSI分支的编码路径
        hr_x1 = self.ReLU(self.hr_inc(hr_msi))  # (1, 64, 240, 240)
        hr_x2 = self.hr_down1(hr_x1)  # (1, 128, 120, 120)
        hr_x3 = self.hr_down2(hr_x2)  # (1, 256, 60, 60)
        hr_x4 = self.hr_down3(hr_x3)  # (1, 512, 30, 30)


        # Lr-HSI分支（调整通道数匹配hr_x4）
        lr_x = self.lr_inc(lr_hsi)  # (1, 512, 30, 30)

        # 在hr_x4和Lr-HSI尺度应用FRM和MDRM模块
        frm_out_1 = self.frm1(hr_x4, lr_x)         # (1, 512, 30, 30)
        fused_feat_1, cp_recon_1 = self.mdrm1(frm_out_1, lr_x)     # (1, 512, 30, 30)
        fused_feat_1 = self.Sigmoid(fused_feat_1)     # (1, 512, 30, 30)
        hr_x3 = self.Sigmoid(hr_x3)  # (1, 256, 60, 60)

        # =========================打印调试信息=================================
        # print("fused_feat", fused_feat.shape)
        # print("hr_x3", hr_x3.shape)
        # print("hr_x2", hr_x2.shape)
        # print("hr_x1", hr_x1.shape)
        # =========================打印调试信息=================================


        # 解码器路径
        x1 = self.up1(fused_feat_1, hr_x3)  # (1, 256, 60, 60)
        # 在hr_x3和x1应用FRM和MDRM模块
        frm_out_2 = self.frm2(hr_x3, x1)  # (1, 256, 60, 60)
        fused_feat_2, cp_recon_2 = self.mdrm2(frm_out_2, x1)     # (1, 256, 60, 60)
        fused_feat_2 = self.Sigmoid(fused_feat_2)  # (1, 256, 60, 60)

        x2 = self.up2(fused_feat_2, hr_x2)   # (1, 128, 120, 120)
        # 在hr_x2和x2应用FRM和MDRM模块
        frm_out_3 = self.frm3(hr_x2, x2)       # (1, 128, 120, 120)
        fused_feat_3, cp_recon_3 = self.mdrm3(frm_out_3, x2)     # (1, 128, 120, 120)
        fused_feat_3 = self.Sigmoid(fused_feat_3)  # (1, 128, 120, 120)

        x3 = self.up3(fused_feat_3, hr_x1)

        # 在hr_x1和x3应用FRM和MDRM模块
        frm_out_4 = self.frm4(hr_x1, x3)       # (1, 64, 240, 240)
        fused_feat_4, cp_recon_4 = self.mdrm4(frm_out_4, x3)     # (1, 64, 240, 240)
        fused_feat_4 = self.Sigmoid(fused_feat_4)  # (1, 64, 240, 240)

        x = self.up4(fused_feat_4)

        y1 = self.conv1(cp_recon_1)      # cp_recon_1(1,512,30,30)---->y1(1, 64, 30, 30)
        y2 = self.conv2(cp_recon_2)      # cp_recon_2(1,256,60,60)---->y2(1, 64, 60, 60)
        y3 = self.conv3(cp_recon_3)      # cp_recon_3(1,128,120,120)----->y3(1, 64, 120, 120)
        y2_up = F.interpolate(y1, size=y2.shape[2:], mode='bilinear', align_corners=True)        # (1, 64, 60, 60)
        y2_fuse = self.fuse3(torch.cat([y2, y2_up], dim=1)) + y2    # (1, 64, 60, 60)

        y3_up = F.interpolate(y2_fuse, size=y3.shape[2:], mode='bilinear', align_corners=True)      # (1, 64, 120, 120)
        y3_fuse = self.fuse2(torch.cat([y3, y3_up], dim=1)) + y3    # (1, 64, 120, 120)

        y4_up = F.interpolate(y3_fuse, size=cp_recon_4.shape[2:], mode='bilinear', align_corners=True)   # (1, 64, 240, 240)
        y4_fuse = self.fuse1(torch.cat([cp_recon_4, y4_up], dim=1)) + cp_recon_4      # (1, 64, 240, 240)

        y = self.out(y4_fuse)      # (1, 64, 240, 240)
        # =========================打印调试信息=================================
        # print("x1", x1.shape)
        # print("x2", x2.shape)
        # print("x3", x3.shape)
        # print("x", x.shape)
        # =========================打印调试信息=================================



        # 将CP重构特征和融合特征映射到Hr-HSI的维度
        mapped_cp_recon = self.cp_recon_mapping(y)    # (1,54,240,240)
        mapped_fused_feat = self.fused_feat_mapping(x)

        return mapped_cp_recon, mapped_fused_feat
