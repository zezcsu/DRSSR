#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

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
import torch.nn as nn
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

from dataset.realesrgan import RealESRGAN_degradation
from utils.img_utils import convert_image_to_fn
from utils.utils import (
    exists,
    instantiate_from_config
)
from utils.cal_mask_torch import cal_detection_mask,DiceBCELoss
if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.31.0.dev0")

logger = get_logger(__name__)
if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False

from torchvision.utils import save_image
from PIL import Image
import numpy as np


def save_batch_grayscale(tensor, output_dir, prefix="img", format="png"):
    """
    保存多批次单通道 Tensor 为图片

    Args:
        tensor (torch.Tensor): 输入 Tensor，形状可以是：
            - (B, 1, H, W)  # 带通道维度
            - (B, H, W)     # 无通道维度
        output_dir (str): 输出目录
        prefix (str): 文件名前缀（默认 "img"）
        format (str): 图片格式（默认 "png"）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 确保输入是 CPU Tensor
    tensor = tensor.cpu().detach()

    # 统一转换为 (B, H, W) 格式
    if tensor.dim() == 4:
        tensor = tensor.squeeze(1)  # 移除通道维度 (B,1,H,W) -> (B,H,W)

    # 遍历批次并保存
    for i in range(tensor.size(0)):
        img_np = tensor[i].numpy()  # (H, W)
        img_np = (img_np * 255).astype(np.uint8)  # [0,1] -> [0,255]

        filename = f"{prefix}_{i}.{format}"
        filepath = os.path.join(output_dir, filename)

        Image.fromarray(img_np).save(filepath)

    print(f"Saved {tensor.size(0)} images to {output_dir}")

def save_binary_masks(mask_tensor, output_dir, prefix="mask"):
    """
    将 (B, 1, H, W) 的二值 mask tensor 保存为多个 PNG 图像

    参数:
        mask_tensor: 输入 tensor，形状为 (B, 1, H, W)，值应为 0 和 1
        output_dir: 输出目录路径
        prefix: 生成文件名的前缀
    """
    # 检查输入 tensor 的维度
    if mask_tensor.dim() != 4 or mask_tensor.size(1) != 1:
        raise ValueError("输入 tensor 的形状应为 (B, 1, H, W)")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 将 tensor 转换为 numpy 数组并移除单通道维度
    masks = mask_tensor.squeeze(1).cpu().numpy()  # 形状变为 (B, H, W)

    # 遍历每个 mask
    for i in range(masks.shape[0]):
        # 获取当前 mask (H, W)
        mask = masks[i]

        # 将值从 [0,1] 转换为 [0,255] 的 uint8
        mask_image = (mask * 255).astype('uint8')

        # 创建 PIL 图像
        img = Image.fromarray(mask_image, mode='L')  # 'L' 表示灰度模式

        # 保存图像
        filename = f"{prefix}_{i:04d}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)

        print(f"已保存: {filepath}")

def log_validation(srmodel, diffmodel, args, accelerator, step):

    logger.info("Running validation... ")

    image_logs = []
    diffmodel.eval()

    diff_dir = os.path.join(args.output_dir, "diff", f"diff-{step}")
    os.makedirs(diff_dir, exist_ok=True)
    for validation_image in args.validation_image:
        shutil.copy(validation_image, diff_dir)

        validation_image = Image.open(validation_image).convert("RGB")
        transform = transforms.ToTensor()
        validation_image = transform(validation_image).unsqueeze(0)
        validation_image = validation_image.to(accelerator.device)

        with torch.no_grad():
            diff = diffmodel(validation_image)

        image_name = os.path.splitext(os.path.basename(diff_dir))[0]
        #diff_path = os.path.join(diff_dir, f"{image_name}_diff.png")

        save_batch_grayscale(torch.clamp(diff,0,1), diff_dir)

        #save_binary_masks(diff, diff_dir)
        image_logs.append(
            {"diff": diff}
        )

    diffmodel.train()
    return image_logs


def save_models(diffmodel, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    torch.save(diffmodel.state_dict(), os.path.join(output_dir, 'diffmodel.pth'))

class LossRecorder:
    r"""
    Class to record better losses.
    """

    def __init__(self, gamma=0.9, max_window=None):
        self.losses = []
        self.gamma = gamma
        self.ema = 0
        self.t = 0
        self.max_window = max_window

    def add(self, *, loss: float) -> None:
        self.losses.append(loss)
        if self.max_window is not None and len(self.losses) > self.max_window:
            self.losses.pop(0)
        self.t += 1
        ema = self.ema * self.gamma + loss * (1 - self.gamma)
        ema_hat = ema / (1 - self.gamma ** self.t) if self.t < 500 else ema
        self.ema = ema_hat

    def moving_average(self, *, window: int) -> float:
        if len(self.losses) < window:
            window = len(self.losses)
        return sum(self.losses[-window:]) / window


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a Diff model training script.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )

    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=25)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="adafactor",
        help="The optimizer type to use. Choose between ['adamw', 'adafactor']",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--adafactor_relative_step", type=bool, default=False, help="Relative step size for Adafactor.")
    parser.add_argument("--adafactor_scale_parameter", type=bool, default=False, help="Scale the initial parameter.")
    parser.add_argument("--adafactor_warmup_init", type=bool, default=False, help="Warmup the initial parameter.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
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
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="sd_xl_train_diffmodel",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
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

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")
    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


def get_train_dataset(args, accelerator):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    while True:
        try:
            if args.dataset_name is not None:
                # Downloading and loading a dataset from the hub.
                dataset = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    cache_dir=args.cache_dir,
                )
            else:
                if args.train_data_dir is not None:
                    dataset = load_dataset(
                        path="imagefolder",
                        data_dir=args.train_data_dir,
                        cache_dir=args.cache_dir,
                    )
                # See more about loading custom images at
                # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script
            break
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            logger.error("Retry...")
            continue

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )


    with accelerator.main_process_first():
        train_dataset = dataset["train"].shuffle(seed=args.seed)
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))
    return train_dataset


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def prepare_train_dataset(dataset, accelerator, convert_image_to="RGB", center_crop=False, random_flip=False):
    degradation = RealESRGAN_degradation(device='cpu')

    maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()
    crop_preproc = transforms.Compose([
        transforms.Lambda(maybe_convert_fn),
        transforms.CenterCrop(args.resolution) if center_crop else transforms.RandomCrop(args.resolution),
        transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
    ])
    #controlnetxt进行0.5norm,PASD没有,是一个可更改的点，但是之前很有可能是其引发的错误
    img_preproc = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    def preprocess_train(examples):
        if crop_preproc is not None:
            images = [crop_preproc(image) for image in examples[args.image_column]]

            hrs, lrs = zip(*[degradation.degrade_process(np.asarray(image) / 255.,
                                                         resize_bak=False) for image in images])
            # 将结果转换为列表（zip返回的是元组）
            hrs = list(hrs)
            lrs = list(lrs)

        examples["conditioning_pixel_values"] = lrs
        examples["pixel_values"] = hrs

        return examples

    with accelerator.main_process_first():
        dataset = dataset.with_transform(preprocess_train)

    return dataset

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()


    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
    }


def downsample_binary_tensor(x, scale_factor=8):
    """
    下采样二进制tensor(只包含0和1)
    参数:
        x: 输入tensor (2D或更高维)
        scale_factor: 下采样比例
    返回:
        下采样后的tensor
    """
    # 确保输入是二进制tensor
    assert torch.all((x == 0) | (x == 1)), "输入tensor必须只包含0和1"

    # 添加batch和channel维度(如果是2D tensor)
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(0)

    # 使用最大池化
    pool = nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor)
    downsampled = pool(x.float())

    # 移除添加的维度并转换回原类型
    downsampled = downsampled.squeeze()
    return downsampled.byte() if x.dtype == torch.uint8 else downsampled
import torch
from PIL import Image
import os



def main(args):

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load first stage model
    cfg = OmegaConf.load(args.first_stage_model_config)
    hat: HAT = instantiate_from_config(cfg.model.hat)
    loadnet = torch.load(cfg.train.hat_path, map_location="cpu")
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    hat.load_state_dict(loadnet[keyname], strict=True)
    hat.eval().to(accelerator.device)

    for p in hat.parameters():
        p.requires_grad = False
    if accelerator.is_main_process:
        logger.info(f"load HAT from {cfg.train.hat_path}")

    diffmodel: UNetResNet50 = instantiate_from_config(cfg.model.unetresnet50)
    if args.resume_from_checkpoint:
        sd = torch.load(cfg.train.unetresnet50_path, map_location="cpu")
        diffmodel.load_state_dict(sd, strict=True)

        if accelerator.is_main_process:
            logger.info(f"load UNetResNet50 from {cfg.train.unetresnet50_path}")
    else:
        logger.info("Initializing UNetResNet50 weights from scratch")
        pass

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        args.learning_rate_controlnet = (
                args.learning_rate_controlnet * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.optimizer_type.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW
        optimizer_kwargs = dict(
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    elif args.optimizer_type.lower() == "adafactor":
        optimizer_class = transformers.optimization.Adafactor
        optimizer_kwargs = dict(
            relative_step=args.adafactor_relative_step,
            scale_parameter=args.adafactor_scale_parameter,
            warmup_init=args.adafactor_warmup_init,
        )
    else:
        raise ValueError(f"Optimizer type {args.optimizer_type} not supported.")

    # Optimizer creation
    diffmodel.train().to(accelerator.device)
    diffmodel.requires_grad_(True)
    params_to_optimize = [{'params': list(diffmodel.parameters()), 'lr': args.learning_rate}]
    logger.info(
        f"Number of trainable parameters in diffmodel: {sum(p.numel() for p in diffmodel.parameters() if p.requires_grad)}")

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        **optimizer_kwargs,
    )

    """
    数据集读取
    """
    train_dataset = get_train_dataset(args, accelerator)

    # Then get the training dataset ready to be passed to the dataloader.
    train_dataset = prepare_train_dataset(train_dataset, accelerator)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
                args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    diffmodel, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        diffmodel, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_image")
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)


    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0


    initial_global_step = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    loss_recorder = LossRecorder(gamma=0.9)

    image_logs = None
    criterion = DiceBCELoss()
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(diffmodel):

                # Convert images to latent space
                pixel_values = ((batch["pixel_values"]+1)/2).to(accelerator.device) # [-1,1]->[0,1]
                controlnet_image = batch["conditioning_pixel_values"].to(accelerator.device)

                sr_image = hat(controlnet_image)

                sr_image = torch.clamp(sr_image, 0, 1)

                masks = cal_detection_mask(sr_image,pixel_values)

                masks = downsample_binary_tensor( masks,8)
                masks = masks.unsqueeze(1)
                #save_binary_masks(masks, 'masks', prefix="mask")
                pre_masks = diffmodel(controlnet_image)

                #可以尝试多种Loss

                loss = criterion(pre_masks, masks)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = []
                    for p in params_to_optimize:
                        params_to_clip.extend(p["params"])
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, "checkpoints", f"checkpoint-{global_step}")
                        save_models(
                            accelerator.unwrap_model(diffmodel),
                            save_path,
                        )
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_image is not None and global_step % args.validation_steps == 0:
                        image_logs = log_validation(
                            srmodel=hat,
                            diffmodel=diffmodel,
                            args=args,
                            accelerator=accelerator,
                            step=global_step,
                        )

            loss = loss.detach().item()
            loss_recorder.add(loss=loss)
            loss_avr: float = loss_recorder.moving_average(window=1000)
            loss_ema: float = loss_recorder.ema
            logs = {"loss/step": loss, 'loss_avr/step': loss_avr, 'loss_ema/step': loss_ema,
                    'lr/step': lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "checkpoints", "final")
        save_models(
            accelerator.unwrap_model(diffmodel),
            save_path,
        )

        # Run a final round of validation.
        image_logs = None
        if args.validation_image is not None:
            image_logs = log_validation(
                srmodel=hat,
                diffmodel=diffmodel,
                args=args,
                accelerator=accelerator,
                step=global_step,
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
