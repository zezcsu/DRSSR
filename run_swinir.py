import argparse
import functools
import gc
import re
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers

from omegaconf import OmegaConf
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from safetensors.torch import load_file, save_file
from pipeline.pipeline_controlnext import StableDiffusionXLControlNeXtPipeline
from models.controlnet import ControlNetModel
from models.unet import UNet2DConditionModel
from dataset.realesrgan import RealESRGAN_degradation
from utils.img_utils import convert_image_to_fn
from utils.utils import (
    exists,
    instantiate_from_config
)

def swinir_inference(img,model,window_size=8):
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
        output = output[:, :, 0:h - mod_pad_h * 4, 0:w - mod_pad_w * 4]
        return output
        
def parse_args(input_args=None):
    
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--first_stage_model_config",
        type=str,
        default="config/SwinIR_B.yaml",
        help=(
            "first super-resolution stage pretrain model(swin, etc.) config"
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default="AIDtest_pro/LR/baseballfield_4_lr.png",
        help=(
            "validation image path"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help=(
            "validation image path"
        ),
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args
    
def eval(args):
    print(args)
    # Load first stage model
    cfg = OmegaConf.load(args.first_stage_model_config)
    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    loadnet = torch.load(cfg.train.swinir_path, map_location="cpu")
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    swinir.load_state_dict(loadnet[keyname], strict=True)

    print(f"load SwinIR from {cfg.train.swinir_path}")
        
    swinir.eval().to(args.device)

    
    validation_image = Image.open(args.validation_image).convert("RGB")
    
    transform = transforms.ToTensor()
    validation_image = transform(validation_image).unsqueeze(0)
    validation_image = validation_image.to(args.device)
    validation_image = swinir_inference(validation_image,swinir,window_size=8)

    output_np = validation_image.cpu().detach().numpy()
    
    # 遍历批次中的每一张图片
    for i in range(output_np.shape[0]):
        # 调整维度顺序为 (h, w, 3)
        img = np.transpose(output_np[i], (1, 2, 0))
        
        print("img.min(),img.max()",img.min(),img.max())
        
        # 缩放到 [0, 255] 并转换为 uint8
        img = (img * 255).astype(np.uint8)
        
        # 使用 PIL 保存图片
        img_pil = Image.fromarray(img)
        img_pil.save(f'output_{i}.png')

if __name__ == "__main__":
    args = parse_args()
    eval(args)