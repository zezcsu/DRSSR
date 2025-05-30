accelerate launch train_diffmodel.py --output_dir "train/example" \
--logging_dir "logs" \
--num_train_epochs 5 \
--resolution 512 \
--dataset_name "data" \
--image_column "image" \
--validation_image "examples/rssr/aid_stadium_202_4.jpg" \
--num_validation_images 1 \
--train_batch_size 4 \
--validation_steps 100 \
--first_stage_model_config "config/SwinIR_B.yaml"
