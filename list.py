from huggingface_hub import  HfApi

# 指定模型仓库 ID（例如 "bert-base-uncased"）
repo_id = "LHRS/RSSR"


# 方法2：使用 HfApi（更灵活）
api = HfApi()
files = api.list_repo_files(repo_id=repo_id)
print("\n文件列表（简洁版）：")
for file in files:
    print(file)