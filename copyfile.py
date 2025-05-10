"""
复制相同名字的图片
"""
import os
import shutil

file_list = "filenames.txt"
source_folder = "data/train"
dest_folder = "img1024"

# 创建目标文件夹（如果不存在）
os.makedirs(dest_folder, exist_ok=True)

with open(file_list, 'r', encoding='utf-8') as f:
    for line in f:
        filename = line.strip()  # 去除换行符和空格
        source_path = os.path.join(source_folder, filename)
        dest_path = os.path.join(dest_folder, filename)
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, dest_path)
            print(f"已复制: {filename}")
        else:
            print(f"文件不存在: {filename}")