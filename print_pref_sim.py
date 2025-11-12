import torch

# path = "outputs/pref_similarity/allenai-OLMo-1B-hf_ultrafeedback_binarized/results_samples.pt"
path = "outputs/pref_similarity/model_epoch_0_Goodreads/results_samples.pt"
# 讀檔（用 CPU 載入，避免需要 GPU）
res = torch.load(path, map_location="cpu")

print("Keys:", list(res.keys()))
for k, v in res.items():
    # v 可能是 tensor 或 list；轉成 Python list 比較好看
    if hasattr(v, "detach"):
        vv = v.detach().cpu()
        head = vv[:5].tolist()
        shape = tuple(vv.shape)
    else:
        head = v[:5]
        shape = (len(v),)
    print(f"\n[{k}] shape={shape}\nhead={head}")
