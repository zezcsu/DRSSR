import os
import PIL
import cv2
import math
import numpy as np
import torch
import torchvision
import imageio
from einops import rearrange


def convert_image_to_fn(img_type, image, minsize=512, eps=0.02):
    width, height = image.size
    if min(width, height) < minsize:
        scale = minsize/min(width, height) + eps
        image = image.resize((math.ceil(width*scale), math.ceil(height*scale)))

    if image.mode != img_type:
        return image.convert(img_type)
    return image