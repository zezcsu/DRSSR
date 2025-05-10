import os
# 修改为镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

snapshot_download(
  repo_id="LHRS/RSSRDATA",
  repo_type="dataset",
  local_dir="img1024/train",
  allow_patterns=["new_train_part_1.tar","new_train_part_17.tar"],
  local_dir_use_symlinks=False,
  resume_download=False,
#   proxies={"https": "http://localhost:7890"}, # clash default port
  max_workers=8
)
