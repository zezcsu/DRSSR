python run_controlnext.py  --pretrained_model_name_or_path "checkpoint/stable-diffusion-xl-refiner-1.0" \
    --unet_model_name_or_path "train/example/checkpoints/checkpoint-10000" \
    --controlnet_model_name_or_path "train/example/checkpoints/checkpoint-10000" \
    --controlnet_scale 1.0 \
    --vae_model_name_or_path "checkpoint/stable-diffusion-xl-refiner-1.0/sdxl-vae-fp16-fix" \
    --validation_prompt "satellite image, clean image, airport, road." \
    --validation_image "examples/rssr/airport_127_128.png" \
    --output_dir "inference" \
    --variant fp16 \
    --num_inference_steps 20 \
    --color_fix \
    --use_safetensors \
    --first_stage_model_config "config/SwinIR_B.yaml" \
    --num_validation_images 1 \

autodl-tmp/ControlNeXt-SDXL-Training/AIDtest_pro/LR/baseballfield_4_lr.png