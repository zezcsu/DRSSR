import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy import ndimage
from torch import Tensor
import numpy as np
import cv2  # 仅用于形态学操作（可替换为kornia）


def gaussian_kernel(kernel_size: int, sigma: float) -> Tensor:
    """生成2D高斯核（PyTorch Tensor版本）"""
    x = torch.arange(kernel_size).float() - kernel_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    kernel = gauss / gauss.sum()
    return torch.outer(kernel, kernel)  # (kernel_size, kernel_size)


def calc_single_artifact_map_torch(
        img: Tensor,
        img2: Tensor,
        window: Tensor,
        padding: int
) -> Tensor:
    """PyTorch版本的单个通道artifact计算（输入范围[0,1]）"""
    constant = (0.03) ** 2  # 从(0.03*255)^2调整为(0.03)^2

    # 使用卷积计算均值
    mu1 = F.conv2d(img, window, padding=padding)
    mu2 = F.conv2d(img2, window, padding=padding)

    # 计算方差
    sigma1_sq = F.conv2d(img ** 2, window, padding=padding) - mu1 ** 2
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=padding) - mu2 ** 2

    # 对比度指标（公式与原始版本一致，仅常数项调整）
    return (2 * torch.sqrt(sigma1_sq + 1e-8) * torch.sqrt(sigma2_sq + 1e-8) + constant) / \
           (sigma1_sq + sigma2_sq + constant)


def cal_detection_mask(
        img1: Tensor,
        img2: Tensor,
        window_size: int = 11,
        contrast_threshold: float = 0.7,  # 阈值范围仍为[0,1]
        area_threshold: int = 4000,
        morph_kernel_size: int = 5
) -> Tensor:
    """
    输入两个[0,1]范围的Tensor，输出二值化Mask (0或1)

    参数:
        img1 (Tensor): 输入图像1 (B,C,H,W) 或 (C,H,W), 范围[0,1]
        img2 (Tensor): 输入图像2 (相同形状)
        window_size: 高斯窗口大小
        contrast_threshold: 对比度阈值(小于该值视为artifact)
        area_threshold: 最小连通区域面积阈值（像素数）
        morph_kernel_size: 形态学操作核大小

    返回:
        Tensor: 二值Mask (0或1), 形状与输入相同
    """
    # 输入检查
    assert img1.shape == img2.shape, "输入Tensor形状必须相同"
    assert img1.max() <= 1.0 and img1.min() >= 0.0, "输入范围应为[0,1]"
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)  # 添加batch维度
        img2 = img2.unsqueeze(0)

    B, C, H, W = img1.shape
    device = img1.device

    # 生成高斯核
    kernel = gaussian_kernel(window_size, 1.5).to(device)
    window = kernel.view(1, 1, window_size, window_size).repeat(C, 1, 1, 1)  # (C,1,k,k)
    padding = window_size // 2

    # 计算各通道的artifact map
    artifact_maps = []
    for c in range(C):
        chan_map = calc_single_artifact_map_torch(
            img1[:, c:c + 1, :, :],
            img2[:, c:c + 1, :, :],
            window[c:c + 1, ...],
            padding
        )
        artifact_maps.append(chan_map)

    artifact_map = torch.mean(torch.cat(artifact_maps, dim=1), dim=1, keepdim=True)  # (B,1,H,W)

    # 生成初步Mask (0或1)
    mask = (artifact_map < contrast_threshold).float()  # (B,1,H,W)

    # 形态学操作 (需要转换为0-255范围的numpy数组)
    final_masks = []
    for b in range(B):
        single_mask = (mask[b, 0, ...].cpu().numpy() * 255).astype(np.uint8)  # (H,W) 0-255

        # 形态学操作
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        erosion = cv2.erode(single_mask, kernel, iterations=1)
        dilation = cv2.dilate(erosion, kernel, iterations=3)
        closed = ndimage.binary_fill_holes(dilation, structure=np.ones((3, 3))).astype(np.uint8)

        # kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        # closed = cv2.morphologyEx(single_mask, cv2.MORPH_CLOSE, kernel)

        # 连通区域过滤
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        output = np.zeros_like(closed)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= area_threshold:
                output[labels == i] = 255

        final_masks.append(torch.from_numpy(output / 255.0).to(device))  # 转回[0,1]

    return torch.stack(final_masks, dim=0).unsqueeze(1)  # (B,1,H,W)


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        bce = nn.functional.binary_cross_entropy(pred, target)
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = 1 - (2. * intersection + 1) / (pred.sum() + target.sum() + 1)
        return bce + dice

