import torch
import torch.nn as nn
import torch.nn.functional as F
from grpc.beta.implementations import insecure_channel
from numpy.ma.core import identity


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class NoiseResNet(nn.Module):
    def __init__(self,
                 in_channels=4,
                 out_channels=4
                ):
        super(NoiseResNet, self).__init__()

        # 输入层 (4 -> 128)
        self.conv_in1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn_in1 = nn.BatchNorm2d(64)
        self.conv_in2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn_in2 = nn.BatchNorm2d(128)

        # 主干网络 (6个残差块，全部128通道)
        self.res_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )

        # 输出层 (128 -> 64 -> 4)
        self.conv_mid = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn_mid = nn.BatchNorm2d(64)
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x[:, :4, :, :]
        # 输入处理
        x = F.relu(self.bn_in2(self.conv_in2(self.bn_in1(self.conv_in1(x)))))

        # 残差块处理
        x = self.res_blocks(x)

        # 输出处理
        x = F.relu(self.bn_mid(self.conv_mid(x)))
        x = self.conv_out(x)+identity
        return x


# 参数计算
model = NoiseResNet()
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params / 1e6:.2f}M")  # 约10.1M