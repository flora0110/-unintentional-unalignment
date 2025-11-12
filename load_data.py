import os
from datasets import load_dataset, DownloadConfig
from datasets.utils.logging import set_verbosity_info, enable_progress_bar

print("setup")
# 開啟詳細日誌 & 進度條
set_verbosity_info()
enable_progress_bar()
print("set cache dir")
cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
print("start download")

dataset_name = "HuggingFaceH4/ultrafeedback_binarized"
ds = load_dataset(
    dataset_name,
    split="train_prefs",
    cache_dir=cache_dir,
    download_config=DownloadConfig(local_files_only=False)  # 需要離線時可改成 True
)