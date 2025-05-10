import torch
import torch.nn as nn


class DoubleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        # 编码器（双分支）
        self.enc1_conv1 = ResidualBlock(in_channels, 64)
        self.enc1_pool = nn.MaxPool2d(2)
        self.enc1_conv2 = ResidualBlock(64, 128)
        self.enc1_pool2 = nn.MaxPool2d(2)

        self.enc2_conv1 = ResidualBlock(in_channels, 64)
        self.enc2_pool = nn.MaxPool2d(2)
        self.enc2_conv2 = ResidualBlock(64, 128)
        self.enc2_pool2 = nn.MaxPool2d(2)

        # 瓶颈层
        self.bottleneck = ResidualBlock(256, 512)

        # 解码器（调整后的结构）
        self.upconv1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec_conv1 = ResidualBlock(512, 256)  # 512 = 256(upconv1) + 128(enc1) + 128(enc2)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec_conv2 = ResidualBlock(256, 128)  # 256 = 128(upconv2) + 64(enc1) + 64(enc2)

        self.final_conv = nn.Conv2d(128, out_channels, 1)

    def forward(self, x1, x2):
        # 编码器分支1
        x1_1 = self.enc1_conv1(x1)  # [B,64,H,W]
        x1_p1 = self.enc1_pool(x1_1)  # [B,64,H/2,W/2]
        x1_2 = self.enc1_conv2(x1_p1)  # [B,128,H/2,W/2]
        x1_p2 = self.enc1_pool2(x1_2)  # [B,128,H/4,W/4]

        # 编码器分支2
        x2_1 = self.enc2_conv1(x2)  # [B,64,H,W]
        x2_p1 = self.enc2_pool(x2_1)  # [B,64,H/2,W/2]
        x2_2 = self.enc2_conv2(x2_p1)  # [B,128,H/2,W/2]
        x2_p2 = self.enc2_pool2(x2_2)  # [B,128,H/4,W/4]

        # 特征融合
        merged = torch.cat([x1_p2, x2_p2], dim=1)  # [B,256,H/4,W/4]
        bottleneck = self.bottleneck(merged)  # [B,512,H/4,W/4]

        # 解码器
        up1 = self.upconv1(bottleneck)  # [B,256,H/2,W/2]
        merge1 = torch.cat([up1, x1_2, x2_2], dim=1)  # 256+128+128=512
        dec1 = self.dec_conv1(merge1)  # [B,256,H/2,W/2]

        up2 = self.upconv2(dec1)  # [B,128,H,W]
        merge2 = torch.cat([up2, x1_1, x2_1], dim=1)  # 128+64+64=256
        dec2 = self.dec_conv2(merge2)  # [B,128,H,W]

        return self.final_conv(dec2) # [B,3,H,W]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # 用于调整输入维度（如果需要）
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        # 保存原始输入

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        identity = out
        out = self.conv2(out)
        out = self.bn2(out)
        

        # 如果输入输出维度不一致（如stride>1或通道数变化），通过1x1卷积调整
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 残差连接
        out = self.relu2(out)
        return out


# 示例用法
if __name__ == "__main__":
    # 超参数
    batch_size = 6
    img_size = 2048
    in_channels = 3
    out_channels = 3

    # 初始化模型
    model = DoubleUNet(in_channels, out_channels)

    # 示例输入
    x1 = torch.randn(batch_size, in_channels, img_size, img_size)
    x2 = torch.randn(batch_size, in_channels, img_size, img_size)

    # 前向传播
    output = model(x1, x2)
    print(f"Input shape: {x1.shape}, {x2.shape}")
    print(f"Output shape: {output.shape}")