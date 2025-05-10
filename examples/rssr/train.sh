accelerate launch train_controlnext.py --pretrained_model_name_or_path "checkpoint/stable-diffusion-xl-refiner-1.0" \
--pretrained_vae_model_name_or_path "checkpoint/sdxl-vae-fp16-fix" \
--variant fp16 \
--use_safetensors \
--output_dir "train_600_1000_STEPS_1GPU/example" \
--logging_dir "logs" \
--num_train_epochs 5 \
--resolution 512 \
--gradient_checkpointing \
--set_grads_to_none \
--proportion_empty_prompts 0.2 \
--controlnet_scale_factor 1.0 \
--mixed_precision fp16 \
--enable_xformers_memory_efficient_attention \
--dataset_name "data" \
--image_column image \
--caption_column caption \
--validation_prompt "satellite image, airport, plane, road." \
--validation_image "examples/rssr/airport_127_128.png" \
--num_validation_images 4 \
--train_batch_size 32 \
--validation_steps 100 \
--first_stage_model_config "config/SwinIR_B.yaml" \
--cache_dir cache \
# --pretrained_unet_model_name_or_path "train_200_1000_STEPS_0_5000_1/example/checkpoints/checkpoint-5000/unet_weight_increasements.safetensors" \
# --controlnet_model_name_or_path "train_200_1000_STEPS_0_5000_1/example/checkpoints/checkpoint-5000/controlnet.safetensors"