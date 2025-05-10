# Copyright 2023 The HuggingFace Team. All rights reserved.
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
# limitations under the License.
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.resnet import Downsample2D, ResnetBlock2D
from einops import rearrange


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class ControlNetOutput(BaseOutput):
    """
    The output of [`ControlNetModel`].

    Args:
        down_block_res_samples (`tuple[torch.Tensor]`):
            A tuple of downsample activations at different resolutions for each downsampling block. Each tensor should
            be of shape `(batch_size, channel * resolution, height //resolution, width // resolution)`. Output can be
            used to condition the original UNet's downsampling activations.
        mid_down_block_re_sample (`torch.Tensor`):
            The activation of the midde block (the lowest sample resolution). Each tensor should be of shape
            `(batch_size, channel * lowest_resolution, height // lowest_resolution, width // lowest_resolution)`.
            Output can be used to condition the original UNet's middle block activation.
    """

    down_block_res_samples: Tuple[torch.Tensor]
    mid_block_res_sample: torch.Tensor


class Block2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
    ) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        output_states = ()

        for resnet in zip(self.resnets):
            hidden_states = resnet(hidden_states, temb)
            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class IdentityModule(nn.Module):
    def __init__(self):
        super(IdentityModule, self).__init__()

    def forward(self, *args):
        if len(args) > 0:
            return args[0]
        else:
            return None


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: Optional[int] = None,
                 stride=1,
                 conv_shortcut: bool = False,
                 dropout: float = 0.0,
                 temb_channels: int = 512,
                 groups: int = 32,
                 groups_out: Optional[int] = None,
                 pre_norm: bool = True,
                 eps: float = 1e-6,
                 non_linearity: str = "swish",
                 skip_time_act: bool = False,
                 time_embedding_norm: str = "default",  # default, scale_shift, ada_group, spatial
                 kernel: Optional[torch.FloatTensor] = None,
                 output_scale_factor: float = 1.0,
                 use_in_shortcut: Optional[bool] = None,
                 up: bool = False,
                 down: bool = False,
                 conv_shortcut_bias: bool = True,
                 conv_2d_out_channels: Optional[int] = None,):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=3 if stride != 1 else 1,
                          stride=stride,
                          padding=1 if stride != 1 else 0,
                          bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x, *args):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Block2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            # in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                # ResnetBlock2D(
                #     in_channels=in_channels,
                #     out_channels=out_channels,
                #     temb_channels=temb_channels,
                #     eps=resnet_eps,
                #     groups=resnet_groups,
                #     dropout=dropout,
                #     time_embedding_norm=resnet_time_scale_shift,
                #     non_linearity=resnet_act_fn,
                #     output_scale_factor=output_scale_factor,
                #     pre_norm=resnet_pre_norm,
                BasicBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                ) if i == num_layers - 1 else \
                IdentityModule()
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    # Downsample2D(
                    #     out_channels,
                    #     use_conv=True,
                    #     out_channels=out_channels,
                    #     padding=downsample_padding,
                    #     name="op",
                    # )
                    BasicBlock(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        temb_channels=temb_channels,
                        stride=2,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=dropout,
                        time_embedding_norm=resnet_time_scale_shift,
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                        pre_norm=resnet_pre_norm,
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
    ) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        output_states = ()

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class ControlProject(nn.Module):
    def __init__(self, num_channels, scale=8, is_empty=False) -> None:
        super().__init__()
        assert scale and scale & (scale - 1) == 0
        self.is_empty = is_empty
        self.scale = scale
        if not is_empty:
            if scale > 1:
                self.down_scale = nn.AvgPool2d(scale, scale)
            else:
                self.down_scale = nn.Identity()
            self.out = nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, bias=False)
            for p in self.out.parameters():
                nn.init.zeros_(p)

    def forward(
            self,
            hidden_states: torch.FloatTensor):
        if self.is_empty:
            shape = list(hidden_states.shape)
            shape[-2] = shape[-2] // self.scale
            shape[-1] = shape[-1] // self.scale
            return torch.zeros(shape).to(hidden_states)

        if len(hidden_states.shape) == 5:
            B, F, C, H, W = hidden_states.shape
            hidden_states = rearrange(hidden_states, "B F C H W -> (B F) C H W")
            hidden_states = self.down_scale(hidden_states)
            hidden_states = self.out(hidden_states)
            hidden_states = rearrange(hidden_states, "(B F) C H W -> B F C H W", F=F)
        else:
            hidden_states = self.down_scale(hidden_states)
            hidden_states = self.out(hidden_states)
        return hidden_states


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        
        self.conv0 = nn.Conv2d(channel*2, channel, 3, 1, 1)
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, A, B):


        B = self.conv0(torch.cat([A, B], dim=1))
        max_out = self.mlp(self.max_pool(B))
        avg_out = self.mlp(self.avg_pool(B))
        channel_out = self.sigmoid(max_out + avg_out)
        A = channel_out * A

        max_out, _ = torch.max(B, dim=1, keepdim=True)
        avg_out = torch.mean(B, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        A = spatial_out * A
        return A

class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvNet, self).__init__()
        # 定义一层卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # 定义ReLU激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 前向传播
        x = self.conv(x)
        x = self.relu(x)
        return x
        
class ControlNetModel(ModelMixin, ConfigMixin):

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: List[int] = [128, 128],
        out_channels: List[int] = [128, 256],
        groups: List[int] = [4, 8],
        time_embed_dim: int = 256,
        final_out_channels: int = 384,
    ):
        super().__init__()

        self.time_proj = Timesteps(128, True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(128, time_embed_dim)

        self.A_embedding = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(2, 128),
            nn.ReLU(),
        )

        self.B_embedding = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(2, 128),
            nn.ReLU(),
        )
        self.attn0 = CBAMLayer(channel=128)

        self.A_down_res = nn.ModuleList()
        self.A_down_sample = nn.ModuleList()
        self.attn = nn.ModuleList()
        for i in range(len(in_channels)):
            self.A_down_res.append(
                ResnetBlock2D(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    temb_channels=time_embed_dim,
                    groups=groups[i]
                ),
            )
            self.A_down_sample.append(
                Downsample2D(
                    out_channels[i],
                    use_conv=True,
                    out_channels=out_channels[i],
                    padding=1,
                    name="op",
                )
            )
            self.attn.append(
                CBAMLayer(out_channels[i])
            )

        self.B_down_res = nn.ModuleList()
        self.B_down_sample = nn.ModuleList()
        for i in range(len(in_channels)):
            self.B_down_res.append(
                ConvNet(in_channels=in_channels[i], out_channels=out_channels[i])
            )
            self.B_down_sample.append(
                Downsample2D(
                    out_channels[i],
                    use_conv=True,
                    out_channels=out_channels[i],
                    padding=1,
                    name="op",
                )
            )

        self.mid_convs = nn.ModuleList()
        self.mid_convs.append(nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels[-1],
                out_channels=out_channels[-1],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.GroupNorm(8, out_channels[-1]),
            nn.Conv2d(
                in_channels=out_channels[-1],
                out_channels=out_channels[-1],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.GroupNorm(8, out_channels[-1]),
        ))
        self.mid_convs.append(
            nn.Conv2d(
                in_channels=out_channels[-1],
                out_channels=final_out_channels,
                kernel_size=1,
                stride=1,
            ))
        self.scale = 1.0  # nn.Parameter(torch.tensor(1.))

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    # Copied from diffusers.models.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    def forward(
        self,
        sample: torch.FloatTensor,
        mask: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
    ) -> Union[ControlNetOutput, Tuple]:
        
        normalize_to_01(mask)
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size = sample.shape[0]
        timesteps = timesteps.expand(batch_size)
        t_emb = self.time_proj(timesteps)
        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)
        emb_batch = self.time_embedding(t_emb)

        # Repeat the embeddings num_video_frames times
        # emb: [batch, channels] -> [batch * frames, channels]
        emb = emb_batch

        
        sample = self.A_embedding(sample)
        mask = self.B_embedding(mask)
       
        sample = self.attn0(sample, mask)

        for a_res, a_downsample ,b_res, b_downsample, attn in zip(self.A_down_res, self.A_down_sample, self.B_down_res, self.B_down_sample,self.attn):
            sample = a_res(sample, emb)
            sample = a_downsample(sample, emb)
            mask = b_res(mask)
            mask = b_downsample(mask, emb)
            sample = attn(sample, mask)
            #TODO 对于mask是不是太过了，不用残差快，直接用卷积？


        sample = self.mid_convs[0](sample) + sample
        sample = self.mid_convs[1](sample)
        return {
            'out': sample,
            'scale': self.scale,
        }


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

def normalize_to_01(x):
    # 计算最小值和最大值
    x_min = x.min()
    x_max = x.max()
    
    # 避免除零错误（如果 x_max == x_min，说明所有值相同）
    if x_max == x_min:
        return torch.zeros_like(x)
    
    # 线性归一化
    x_normalized = (x - x_min) / (x_max - x_min)
    return x_normalized
