#!/bin/bash

# 定义txt文件路径
IMAGE_LIST_FILE="image_list.txt"  # 替换为你的txt文件路径
PROMPT="satellite image"
OUTPUT_DIR="inference"

# 检查文件是否存在
if [ ! -f "$IMAGE_LIST_FILE" ]; then
    echo "Error: File $IMAGE_LIST_FILE not found!"
    exit 1
fi

# 逐行读取txt文件并运行
while IFS= read -r IMAGE_PATH; do
    # 跳过空行
    if [ -z "$IMAGE_PATH" ]; then
        continue
    fi

    # 检查图片是否存在
    if [ ! -f "$IMAGE_PATH" ]; then
        echo "Warning: Image $IMAGE_PATH not found, skipping..."
        continue
    fi

    echo "Processing image: $IMAGE_PATH"
    
    python run_controlnext.py --pretrained_model_name_or_path "checkpoint/stable-diffusion-xl-refiner-1.0" \
        --unet_model_name_or_path "train/example/checkpoints/checkpoint-10000" \
        --controlnet_model_name_or_path "train/example/checkpoints/checkpoint-10000" \
        --controlnet_scale 1.0 \
        --vae_model_name_or_path "checkpoint/stable-diffusion-xl-refiner-1.0/sdxl-vae-fp16-fix" \
        --validation_prompt "$PROMPT" \
        --validation_image "$IMAGE_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --variant fp16 \
        --color_fix \
        --use_safetensors \
        --first_stage_model_config "config/SwinIR_B.yaml" 

    echo "Completed: $IMAGE_PATH"
    echo "----------------------------------"
done < "$IMAGE_LIST_FILE"

echo "All images processed!"