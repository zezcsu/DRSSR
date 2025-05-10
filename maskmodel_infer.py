import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
from models.unetresnet50 import UNetResNet50
class BinarySegmentationInference:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化分割模型
        :param model_path: 模型权重路径
        :param device: 使用设备 (cuda/cpu)
        """
        self.device = device
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
        self.threshold = 0.3  # 固定阈值设为0.7
    
    def load_model(self, model_path):
        """
        加载预训练模型
        这里需要替换为你实际的模型类
        """
        model = UNetResNet50()
        model.load_state_dict(torch.load(model_path))
        return model
        #raise NotImplementedError("请实现你的模型加载逻辑")
    
    def preprocess(self, image):
        """
        图像预处理
        :param image: PIL Image
        :return: 预处理后的tensor
        """
        image = np.array(image)
        image = image.astype(np.float32) / 255.0  # 归一化到[0,1]
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return image.to(self.device)
    
    def postprocess(self, output, original_size):
        """
        后处理模型输出
        :param output: 模型原始输出
        :param original_size: 原始图像大小 (H,W)
        :return: 二值化分割掩码 (numpy数组)
        """
        probs = output
        # probs = torch.sigmoid(output)
        # probs = F.interpolate(probs, size=original_size, mode='bilinear', align_corners=False)
        mask = (probs > self.threshold).float().squeeze().cpu().numpy()
        return mask
    
    def save_mask(self, mask, save_path):
        """
        保存二值化掩码为图像文件
        :param mask: 二值化掩码 (0和1的数组)
        :param save_path: 保存路径
        """
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        mask_image.save(save_path)
    
    def predict_and_save(self, image_path, output_dir):
        """
        执行预测并保存结果
        :param image_path: 输入图像路径
        :param output_dir: 输出目录
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载和处理图像
        image = Image.open(image_path).convert('RGB')
        original_size = image.size[::-1]  # (H,W)
        input_tensor = self.preprocess(image)
        
        # 推理
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # 后处理
        mask = self.postprocess(output, original_size)
        
        # 保存结果
        filename = os.path.basename(image_path)
        save_path = os.path.join(output_dir, f"mask_{filename}")
        self.save_mask(mask, save_path)
        
        return mask


# 使用示例
if __name__ == '__main__':
    # 初始化推理器
    model_path = 'train_maskmodel/example/checkpoints/checkpoint-7000/diffmodel.pth'
    segmenter = BinarySegmentationInference(model_path)
    
    # 执行预测并保存
    image_path = 'examples/rssr/LoveDa_Test_5805_lr.png'
    output_dir = 'output_masks'
    mask = segmenter.predict_and_save(image_path, output_dir)