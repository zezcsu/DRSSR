import os
import numpy as np
from PIL import Image
from dataset.realesrgan import RealESRGAN_degradation
from torchvision.transforms import ToPILImage

to_pil = ToPILImage()
def tensor_to_image(tensor):
    """将Tensor转换为PIL图像"""
    # 如果Tensor在GPU上，先移到CPU
    if tensor.is_cuda:
        tensor = tensor.cpu()
    # 移除batch维度（如果有）
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    # 转换到[0,255]范围并转为PIL图像
    tensor = tensor.mul(255).clamp(0, 255).byte()
    return to_pil(tensor)

input_folder="AIDtest"
output_folder="AIDtest_pro"
hr_output_folder = os.path.join(output_folder, 'HR')
lr_output_folder = os.path.join(output_folder, 'LR')

# 创建输出文件夹
os.makedirs(hr_output_folder, exist_ok=True)
os.makedirs(lr_output_folder, exist_ok=True)

# 支持的图片格式
valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
degradation = RealESRGAN_degradation(device='cpu')
# 处理每张图片
for filename in os.listdir(input_folder):
    if filename.lower().endswith(valid_extensions):
        # 读取图片
        img_path = os.path.join(input_folder, filename)
        #try:
        image = Image.open(img_path).convert('RGB')
        img_array = np.asarray(image) / 255.  # 转换为[0,1]范围的numpy数组

        # 使用degrade_process处理
        hrs, lrs = degradation.degrade_process(img_array, resize_bak=False)

        # 将处理结果转换回[0,255]范围并保存
        base_name = os.path.splitext(filename)[0]

        # 保存HR图像
        hr_img = tensor_to_image((hrs+1)/2)
        hr_img.save(os.path.join(hr_output_folder, f'{base_name}_hr.png'))

        # 保存LR图像
        lr_img = tensor_to_image(lrs)
        lr_img.save(os.path.join(lr_output_folder, f'{base_name}_lr.png'))

        print(f'Processed: {filename}')

        # except Exception as e:
        #     print(f'Error processing {filename}: {str(e)}')

print('All images processed!')