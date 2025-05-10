import os
# 修改为镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download
#download stable-diffusion-xl-base-1.0
snapshot_download(
  repo_id="stabilityai/stable-diffusion-xl-base-1.0",
  repo_type="model",
  local_dir="checkpoint/stable-diffusion-xl-base-1.0",
  allow_patterns=["scheduler/scheduler_config.json",
                    "tokenizer/merges.txt",
                    "tokenizer/special_tokens_map.json",
                    "tokenizer/tokenizer_config.json",
                    "tokenizer/vocab.json",
                    "tokenizer_2/merges.txt",
                    "tokenizer_2/special_tokens_map.json",
                    'tokenizer_2/tokenizer_config.json',
                    'tokenizer_2/vocab.json',
                    'unet/config.json',
                    'unet/diffusion_pytorch_model.fp16.safetensors',
                    'unet/diffusion_pytorch_model.safetensors',
                    'text_encoder/config.json',
                    'text_encoder/model.fp16.safetensors',
                    'text_encoder/model.safetensors',
                    'text_encoder_2/config.json',
                    'text_encoder_2/model.safetensors',
                    'text_encoder_2/model.fp16.safetensors',
                    'model_index.json',
                 ],
  local_dir_use_symlinks=False,
  resume_download=False,
#   proxies={"https": "http://localhost:7890"}, # clash default port
  max_workers=8
)


#download first stage model 1
snapshot_download(
  repo_id="LHRS/RSSR",
  repo_type="model",
  local_dir="checkpoint/maskmodel",
  allow_patterns=["diffmodel.pth"],
  local_dir_use_symlinks=False,
  resume_download=False,
#   proxies={"https": "http://localhost:7890"}, # clash default port
  max_workers=8
)

#download first stage model 2
snapshot_download(
  repo_id="LHRS/RSSR",
  repo_type="model",
  local_dir="checkpoint/hat",
  allow_patterns=["hatmse400000.pth"],
  local_dir_use_symlinks=False,
  resume_download=False,
#   proxies={"https": "http://localhost:7890"}, # clash default port
  max_workers=8
)

#download sdxl-vae-fp16-fix
snapshot_download(
  repo_id="madebyollin/sdxl-vae-fp16-fix",
  repo_type="model",
  local_dir="checkpoint/sdxl-vae-fp16-fix",
  allow_patterns=["config.json",
                    "diffusion_pytorch_model.safetensors",
                    "sdxl.vae.safetensors",
                    "sdxl_vae.safetensors"],
  local_dir_use_symlinks=False,
  resume_download=False,
#   proxies={"https": "http://localhost:7890"}, # clash default port
  max_workers=8
)






