import os
import torch
import cv2
import gc
import numpy as np
import argparse
from PIL import Image
from utils import preprocess, tools
from utils.wavelet_color_fix import wavelet_color_fix
import torch.nn.functional as F
from omegaconf import OmegaConf
from utils.utils import (
    exists,
    instantiate_from_config
)
from torchvision import transforms
#copied from https://github.com/XPixelGroup/BasicSR/blob/master/inference/inference_swinir.py
def swinir_inference(img,model,args,window_size=8):
    with torch.no_grad():
        # pad input image to be a multiple of window_size
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = img.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        output = model(img)
        _, _, h, w = output.size()
        output = output[:, :, 0:h - mod_pad_h * args.upscale, 0:w - mod_pad_w * args.upscale]
        return output

def log_validation(
    args,
    device='cuda'
):
    pipeline = tools.get_pipeline(
        args.pretrained_model_name_or_path,
        args.unet_model_name_or_path,
        args.controlnet_model_name_or_path,
        vae_model_name_or_path=args.vae_model_name_or_path,
        lora_path=args.lora_path,
        load_weight_increasement=args.load_weight_increasement,
        enable_xformers_memory_efficient_attention=args.enable_xformers_memory_efficient_attention,
        revision=args.revision,
        variant=args.variant,
        hf_cache_dir=args.hf_cache_dir,
        use_safetensors=args.use_safetensors,
        device=device,
        args=args,

    )
    weight_dtype = torch.float32
    if args.variant == "fp16":
        weight_dtype = torch.float16
    elif args.variant == "bf16":
        weight_dtype = torch.bfloat16

    # Load first stage model
    cfg = OmegaConf.load(args.first_stage_model_config)
    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    loadnet = torch.load(cfg.train.swinir_path, map_location="cpu")
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    swinir.load_state_dict(loadnet[keyname], strict=True)
    for p in swinir.parameters():
        p.requires_grad = False
    print(f"load SwinIR from {cfg.train.swinir_path}")

    diffmodel: DoubleUNet = instantiate_from_config(cfg.model.doubleunet)
    loadnet = torch.load(cfg.train.doubleunet_path, map_location="cpu")
    diffmodel.load_state_dict(loadnet, strict=True)
    for p in diffmodel.parameters():
        p.requires_grad = False
    print(f"load DoubleUNet from {cfg.train.doubleunet_path}")

    swinir.eval().to(device)
    diffmodel.eval().to(device)


    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    if args.negative_prompt is not None:
        negative_prompts = args.negative_prompt
        assert len(validation_prompts) == len(validation_prompts)
    else:
        negative_prompts = None

    #extractor = preprocess.get_extractor(args.validation_image_processor)

    image_logs = []
    inference_ctx = torch.autocast(device)

    resize_preproc = transforms.Compose([
        transforms.Resize((args.process_size//args.upscale), interpolation=transforms.InterpolationMode.BILINEAR)])
    
    save_dir_path = os.path.join(args.output_dir, "eval_img")
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    for i, (validation_prompt, validation_image) in enumerate(zip(validation_prompts, validation_images)):
        
        filename = os.path.splitext(os.path.basename(validation_image))[0]
        validation_image = Image.open(validation_image).convert("RGB")
        log_validation_image = validation_image.copy()
        log_validation_image = log_validation_image.resize(
            ( log_validation_image.size[0] * 4, log_validation_image.size[1] * 4),
            resample=Image.Resampling.LANCZOS  # 最高质量插值
        )

        ori_width, ori_height = validation_image.size
        if min(validation_image.size) < (args.process_size//args.upscale):
            validation_image = resize_preproc(validation_image)
         
        validation_image = validation_image.resize(
            (validation_image.size[0] // 2 * 2, validation_image.size[1] // 2 * 2))


        # if extractor is not None:
        #     validation_image = extractor(validation_image)

        transform = transforms.ToTensor()
        validation_image = transform(validation_image).unsqueeze(0)
        validation_image = validation_image.to(device)
        
        lr = validation_image.clone()
        lr = F.interpolate(
            lr,
            scale_factor=args.upscale,  # 或直接指定尺寸 mode='bilinear', align_corners=False
            mode='bicubic'  # 可选 'nearest', 'bicubic' 等
        )
        lr = torch.clamp(lr, 0, 1)
        validation_image = swinir_inference(validation_image, swinir, args, window_size=8)

        diff = diffmodel(lr, validation_image)

        validation_image = validation_image.clamp(lr, 0, 1)
        #print("validation_image",validation_image.max(),validation_image.min())
        validation_image.to(device, dtype=weight_dtype)
        diff = diff.to(device, dtype=weight_dtype)
        images = []


        negative_prompt = negative_prompts[i] if negative_prompts is not None else None
        for _ in range(args.num_validation_images):
            with inference_ctx:
                #print(" validation_image.shape", validation_image.shape)
                image = pipeline(
                    image=validation_image,
                    diff=diff,
                    prompt=validation_prompt,
                    controlnet_image=validation_image,
                    controlnet_scale=args.controlnet_scale,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                    negative_prompt=negative_prompt,
                    width= validation_image.shape[-1],
                    height= validation_image.shape[-2],
                    aesthetic_score=args.aesthetic_score,
                    guidance_scale=0,
                ).images[0]
            if args.color_fix:
                image = wavelet_color_fix(image, log_validation_image)
            image = image.resize((ori_width * args.upscale, ori_height *  args.upscale))
            file_path = os.path.join(save_dir_path, filename+".png")
            image.save(file_path) 
            
            
    #         images.append(image)

    #     image_logs.append(
    #         {"validation_image": log_validation_image, "images": images, "validation_prompt": validation_prompt}
    #     )


    # for i, log in enumerate(image_logs):
    #     images = log["images"]
    #     validation_prompt = log["validation_prompt"]
    #     validation_image = log["validation_image"]

    #     formatted_images = []
    #     formatted_images.append(np.asarray(validation_image))
    #     for image in images:
    #         formatted_images.append(np.asarray(image))
    #     formatted_images = np.concatenate(formatted_images, 1)

    #     for j, validation_image in enumerate(images):
    #         file_path = os.path.join(save_dir_path, filename+".png")
    #         #file_path = os.path.join(save_dir_path, "image_{}-{}.png".format(i, j))
    #         validation_image = np.asarray(validation_image)
    #         validation_image = cv2.cvtColor(validation_image, cv2.COLOR_BGR2RGB)
    #         cv2.imwrite(file_path, validation_image)
    #         print("Save images to:", file_path)

    #     # file_path = os.path.join(save_dir_path, "image_{}.png".format(i))
    #     # formatted_images = cv2.cvtColor(formatted_images, cv2.COLOR_BGR2RGB)
    #     # print("Save images to:", file_path)
    #     # cv2.imwrite(file_path, formatted_images)

    gc.collect()
    if str(device) == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return image_logs


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--unet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained unet model or subset"
    )
    parser.add_argument(
        "--vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained vae model or subset"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to lora"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--use_safetensors",
        action="store_true",
        help="Whether or not to use safetensors to load the pipeline.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--process_size", type=int, default=512, help="minimal input size for processing")  # 512?
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--validation_image_processor",
        type=str,
        default=None,
        choices=["canny"],
        help="The type of image processor to use for the validation images.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of inference steps for the diffusion model",
    )
    parser.add_argument(
        "--controlnet_scale",
        type=float,
        default=1.0,
        help="Scale of the controlnet",
    )
    parser.add_argument(
        "--load_weight_increasement",
        action="store_true",
        help="Only load weight increasement",
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default=None,
        help="Path to the cache directory for huggingface datasets and models.",
    )
    parser.add_argument("--latent_tiled_size", type=int, default=180, help="unet latent tile size for saving GPU memory") # for 24G
    parser.add_argument("--latent_tiled_overlap", type=int, default=8, help="unet lantent overlap size for saving GPU memory") # for 24G
    parser.add_argument("--upscale", type=int, default=4, help="upsampling scale")
    parser.add_argument(
        "--color_fix",
        action="store_true",
        help="wavelet color fix",
    )
    parser.add_argument(
        "--aesthetic_score",
        type=str,
        default=8,
        help=(
            "aesthetic score of training image"
        ),
    )
    parser.add_argument(
        "--first_stage_model_config",
        type=str,
        default=None,
        help=(
            "first super-resolution stage pretrain model(swin, etc.) config"
        ),
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    # if args.width is not None and args.width % 8 != 0:
    #     raise ValueError(
    #         "`--width` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
    #     )
    # if args.height is not None and args.height % 8 != 0:
    #     raise ValueError(
    #         "`--height` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
    #     )

    return args


if __name__ == "__main__":
    args = parse_args()

    import glob

    folder_path = "RS_C11_Database_pro/LR"
    image_files = glob.glob(os.path.join(folder_path, "*.[pj][np]g")) + \
              glob.glob(os.path.join(folder_path, "*.bmp")) + \
              glob.glob(os.path.join(folder_path, "*.tif*"))

    # 仅保留文件名（去掉路径）
    image_files = [os.path.basename(f) for f in image_files]
    image_paths = [os.path.join(folder_path, f) for f in image_files]
    args.validation_image =image_paths
    
    log_validation(args)
