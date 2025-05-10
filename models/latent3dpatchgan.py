import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

class Latent3DPatchGAN(nn.Module):
    def __init__(self, in_channels=4, ndf=64, n_layers=3, use_sn=True):
        """
        参数:
            in_channels: 输入通道数 (4)
            ndf: 初始特征维度
            n_layers: 卷积层数
            use_sn: 是否使用谱归一化
        """
        super().__init__()

        layers = []

        # 第一层: [B,4,64,64] -> [B,ndf,32,32]
        layers += self._make_layer(in_channels, ndf, kernel_size=4, stride=2, padding=1,
                                   use_sn=use_sn, first_layer=True)

        # 中间层: 逐步下采样
        for i in range(n_layers - 1):
            in_feat = ndf * min( 2**i, 8)
            out_feat = ndf * min( 2**( i +1), 8)
            layers += self._make_layer(in_feat, out_feat, kernel_size=4, stride=2,
                                       padding=1, use_sn=use_sn)

        # 最后一层: [B,ndf*8,16,16] -> [B,1,16,16] (保持空间尺寸)
        final_conv = nn.Conv2d(ndf * min( 2**(n_layers -1), 1, kernel_size=4, stride=1, padding='same'))
        if use_sn:
            final_conv = spectral_norm(final_conv)
        layers += [
            final_conv,
            # 不添加激活函数，直接输出logits
        ]

        self.model = nn.Sequential(*layers)

    def _make_layer(self, in_feat, out_feat, kernel_size, stride, padding, use_sn, first_layer=False):
        layer = []
        conv = nn.Conv2d(in_feat, out_feat, kernel_size, stride, padding)
        if use_sn:
            conv = spectral_norm(conv)
        layer.append(conv)

        if not first_layer:  # 第一层不加IN
            layer.append(nn.InstanceNorm2d(out_feat))

        layer.append(nn.LeakyReLU(0.2, inplace=True))
        return layer

    def forward(self, z):
        """
        输入:
            z: 潜在编码 [B,4,64,64]
        输出:
            patch_logits: 判别结果 [B,1,16,16]
        """
        return self.model(z)  # 输出未经过sigmoid

def soft_labels(shape, device, real_noise=0.05, fake_noise=0.05):
    """
    生成软标签:
    - 真实标签: 0.9 ± random_noise (原1.0)
    - 生成标签: 0.1 ± random_noise (原0.0)
    """
    real_label = torch.full(shape, 0.9, device=device) + torch.rand(shape, device=device) * real_noise
    fake_label = torch.full(shape, 0.1, device=device) + torch.rand(shape, device=device) * fake_noise
    return real_label, fake_label


def adversarial_loss(discriminator, z_pred, z_real):
    # 生成软标签
    b, _, h, w = z_pred.shape
    real_label, fake_label = soft_labels((b, 1, h // 4, w // 4), z_pred.device)  # 匹配输出16x16

    # 判别器损失
    real_logits = discriminator(z_real)
    fake_logits = discriminator(z_pred.detach())

    loss_D_real = F.binary_cross_entropy_with_logits(real_logits, real_label)
    loss_D_fake = F.binary_cross_entropy_with_logits(fake_logits, fake_label)
    loss_D = (loss_D_real + loss_D_fake) * 0.5

    return loss_D

def generator_loss(discriminator,z_pred):
    pred_logits = discriminator(z_pred)  # [B,1,16,16]
    loss = F.binary_cross_entropy_with_logits(
        pred_logits,
        torch.ones_like(pred_logits)  # 生成器希望判别器输出"真"
    )
    return loss