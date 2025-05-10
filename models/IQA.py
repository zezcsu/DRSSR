import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 卷积步长 (默认为1)
            downsample: 用于匹配维度的下采样函数 (默认为None)
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:

            self.conv0 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1,  bias=False
                )
        else:
            self.conv0 = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.conv0 is not None:
            identity = self.conv0(x)

        out += identity  # 残差连接
        out = self.relu(out)

        return out

class DoubleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        # 编码器（双分支）

        # SR 分支
        self.enc1_conv1 = ResidualBlock(in_channels, 64)
        self.enc1_conv2 = ResidualBlock(64, 128)

        # LR 分支
        self.enc2_conv1 = ResidualBlock(in_channels, 64)
        self.enc2_conv2 = ResidualBlock(64, 128)

        # 瓶颈层
        self.bottleneck = ResidualBlock(256, 256)

        # 解码器（调整后的结构）

        self.dec_conv1 = ResidualBlock(512, 128)  # 512 = 256(upconv1) + 128(enc1) + 128(enc2)

        self.dec_conv2 = ResidualBlock(256, 128)  # 256 = 128(upconv2) + 64(enc1) + 64(enc2)

        self.final_conv = nn.Conv2d(128, out_channels, 1)

    def forward(self, x1, x2):
        # 编码器分支1
        x1_1 = self.enc1_conv1(x1)  # [B,64,H,W]

        x1_2 = self.enc1_conv2(x1_1)  # [B,128,H/2,W/2]


        # 编码器分支2
        x2_1 = self.enc2_conv1(x2)  # [B,64,H,W]
        x2_2 = self.enc2_conv2(x2_1)  # [B,128,H/2,W/2]

        # 特征融合
        merged = torch.cat([x1_2, x2_2], dim=1)  # [B,256,H/4,W/4]
        bottleneck = self.bottleneck(merged)  # [B,512,H/4,W/4]

        # 解码器
        up1 = self.upconv1(bottleneck)  # [B,256,H/2,W/2]
        merge1 = torch.cat([up1, x1_2, x2_2], dim=1)  # 256+128+128=512
        dec1 = self.dec_conv1(merge1)  # [B,256,H/2,W/2]

        up2 = self.upconv2(dec1)  # [B,128,H,W]
        merge2 = torch.cat([up2, x1_1, x2_1], dim=1)  # 128+64+64=256
        dec2 = self.dec_conv2(merge2)  # [B,128,H,W]

        return self.final_conv(dec2) # [B,3,H,W]

class MYModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.ex1_conv1 = nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.ex1_conv2 = nn.Conv2d(
            32, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.ex2_conv1 = nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.ex2_pool1 = nn.MaxPool2d(2)
        self.ex2_conv2 = nn.Conv2d(
            32, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.ex2_pool2 = nn.MaxPool2d(2)
        self.doubleUNet = DoubleUNet(64, 64)

        #score output
        self.dec_conv1 = ResidualBlock(512, 128)  # 512 = 256(upconv1) + 128(enc1) + 128(enc2)