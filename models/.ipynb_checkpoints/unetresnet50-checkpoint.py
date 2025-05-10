import torch
import torch.nn as nn
from torchvision import models


class DecoderBlock(nn.Module):
    """Upsample + DoubleConv + Skip Connection"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)

        x = torch.cat([x, skip], dim=1)  # Channel-wise concatenation
        return self.conv(x)


class UNetResNet50(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # Encoder: ResNet50 (移除最后的全连接层)
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # 保留到layer4

        # Decoder (调整通道数以匹配ResNet50的输出)
        self.decoder1 = DecoderBlock(2048, 1024)  # ResNet50的layer4输出2048通道
        self.decoder2 = DecoderBlock(1024, 512)
        self.decoder3 = DecoderBlock(512, 256)
        self.decoder4 = DecoderBlock(256, 64)

        # Final convolution (输出H/2, W/2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        skips = []
        x = self.encoder[0:3](x)  # Conv1 + BN + ReLU + MaxPool (输出64通道)
        skips.append(x)  # Skip1 (64, H/4, W/4)
        x = self.encoder[3](x)
        x = self.encoder[4](x)  # Layer1 (256, H/4, W/4)
        skips.append(x)  # Skip2
        x = self.encoder[5](x)  # Layer2 (512, H/8, W/8)
        skips.append(x)  # Skip3
        x = self.encoder[6](x)  # Layer3 (1024, H/16, W/16)
        skips.append(x)  # Skip4
        x = self.encoder[7](x)  # Layer4 (2048, H/32, W/32)

        # Decoder
        x = self.decoder1(x, skips[3])  # (1024, H/16, W/16)
        x = self.decoder2(x, skips[2])  # (512, H/8, W/8)
        x = self.decoder3(x, skips[1])  # (256, H/4, W/4)
        x = self.decoder4(x, skips[0])  # (64, H/2, W/2)

        # Final output
        return self.final_conv(x)  # (1, H/2, W/2)


# # 检查参数量
# model = UNetResNet50()
# summary(model, (3, 256, 256))  # 输入(3,256,256)，输出(1,128,128)
# print(f"总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
