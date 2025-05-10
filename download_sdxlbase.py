import os
# 修改为镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

#download stable-diffusion-xl-base-1.0
snapshot_download(
  repo_id="stabilityai/stable-diffusion-xl-base-1.0",
  repo_type="model",
  local_dir="checkpoint/stable-diffusion-xl-base-1.0",
  allow_patterns=[
                    'unet/diffusion_pytorch_model.fp16.safetensors',

                 ],
#   proxies={"https": "http://localhost:7890"}, # clash default port
  max_workers=8
)





