accelerate launch train_srgan_latent.py --pretrained_model_name_or_path "checkpoint/stable-diffusion-xl-refiner-1.0" \
--pretrained_vae_model_name_or_path "checkpoint/sdxl-vae-fp16-fix" \
--output_dir "train_srganmodel_latent/example" \
--logging_dir "logs" \
--num_train_epochs 5 \
--resolution 1024 \
--dataset_name "img1024" \
--image_column "image" \
--validation_image "examples/rssr/aid_stadium_202_4.jpg" \
--num_validation_images 1 \
--train_batch_size 32 \
--validation_steps 100 \
--first_stage_model_config "config/srganResNet.yaml"
