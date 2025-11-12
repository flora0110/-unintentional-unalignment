# import torch

# # path = "outputs/pref_similarity/allenai-OLMo-1B-hf_ultrafeedback_binarized/results_samples.pt"
# path = "outputs/pref_similarity/model_epoch_0_Goodreads/results_samples.pt"
# # 讀檔（用 CPU 載入，避免需要 GPU）
# res = torch.load(path, map_location="cpu")

# print("Keys:", list(res.keys()))
# for k, v in res.items():
#     # v 可能是 tensor 或 list；轉成 Python list 比較好看
#     if hasattr(v, "detach"):
#         vv = v.detach().cpu()
#         head = vv[:5].tolist()
#         shape = tuple(vv.shape)
#     else:
#         head = v[:5]
#         shape = (len(v),)
#     print(f"\n[{k}] shape={shape}\nhead={head}")


# inject_pref_similarity.py
import json
import argparse
from pathlib import Path

import torch

def to_plain_list(x):
    # tensor/list 轉成純 Python list
    if hasattr(x, "detach"):
        return x.detach().cpu().tolist()
    return list(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", required=True,
                    help="原始 train.json（JSON list）")
    ap.add_argument("--prefsim_pt", required=True,
                    help="torch.save 的 results_samples.pt 路徑")
    ap.add_argument("--out_json", default=None,
                    help="輸出檔名（預設與 train 同資料夾，檔名加 _with_prefsim）")
    args = ap.parse_args()

    train_path = Path(args.train_json)
    out_path = Path(args.out_json) if args.out_json else train_path.with_name(train_path.stem + "_with_prefsim.json")

    # 1) 讀 metrics
    res = torch.load(args.prefsim_pt, map_location="cpu")
    sample_indices = to_plain_list(res["sample_indices"])
    ches = to_plain_list(res["ches_scores"])
    ln_ches = to_plain_list(res["ln_ches_scores"])
    inner = to_plain_list(res["last_hidden_embedding_inner_prods"])

    assert len(sample_indices) == len(ches) == len(ln_ches) == len(inner), \
        "樣本數不一致：請檢查 results_samples.pt 內容"

    # 2) 讀 train.json
    with open(train_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    n = len(data)

    # 3) 寫入（不存在就新增欄位；存在則覆寫）
    for i, idx in enumerate(sample_indices):
        if idx < 0 or idx >= n:
            # 略過越界 index（理論上不該發生，除非 train 與 prefsim 的子集/排序不一致）
            continue
        rec = data[idx]
        rec["ches_score"] = float(ches[i])
        rec["ln_ches_score"] = float(ln_ches[i])
        rec["last_hidden_embedding_inner_prod"] = float(inner[i])

    # 4) 輸出
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ 寫入完成：{out_path}")
    print(f"  總樣本數：{n}")
    print(f"  更新樣本數：{len(sample_indices)}")
    # 顯示前 3 筆已更新樣本以確認
    shown = 0
    for i in sample_indices:
        rec = data[i]
        if all(k in rec for k in ["ches_score", "ln_ches_score", "last_hidden_embedding_inner_prod"]):
            print({
                "idx": i,
                "ches_score": rec["ches_score"],
                "ln_ches_score": rec["ln_ches_score"],
                "last_hidden_embedding_inner_prod": rec["last_hidden_embedding_inner_prod"],
                "prompt_head": rec.get("prompt", "")[:40] + "..."
            })
            shown += 1
            if shown >= 3:
                break

if __name__ == "__main__":
    main()
