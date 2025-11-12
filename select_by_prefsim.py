# select_by_prefsim.py
import json
import argparse
from pathlib import Path
import random
import math

METRICS = [
    "ches_score",
    "ln_ches_score",
    "last_hidden_embedding_inner_prod",
]

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("輸入 JSON 需為 list 格式。")
    return data

def filter_has_metric(data, metric):
    out = []
    for i, rec in enumerate(data):
        if metric in rec and isinstance(rec[metric], (int, float)) and not (isinstance(rec[metric], float) and math.isnan(rec[metric])):
            out.append(rec)
    return out

def top_k(data, metric, k):
    # 大到小排序
    sorted_data = sorted(data, key=lambda r: r[metric], reverse=True)
    return sorted_data[:k]

def bottom_k(data, metric, k):
    # 小到大排序
    sorted_data = sorted(data, key=lambda r: r[metric])
    return sorted_data[:k]

def random_k(data, k, seed):
    rng = random.Random(seed)
    if len(data) <= k:
        return list(data)
    return rng.sample(data, k)

def write_json(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def main():
    print("hello?")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True, help="含三個指標欄位的 train_with_prefsim.json")
    ap.add_argument("--out_dir", required=False, default=None, help="輸出資料夾（預設與輸入檔同資料夾）")
    ap.add_argument("--k", type=int, default=100, help="每種集合的樣本數")
    ap.add_argument("--seed", type=int, default=42, help="random 選樣用的種子")
    args = ap.parse_args()

    in_path = Path(args.input_json)
    out_dir = Path(args.out_dir) if args.out_dir else in_path.parent

    data = load_data(in_path)
    total = len(data)
    print(f"Loaded {total} records from: {in_path}")

    for metric in METRICS:
        subset = filter_has_metric(data, metric)
        if not subset:
            print(f"[WARN] 無可用資料含指標欄位：{metric}，略過。")
            continue

        k = min(args.k, len(subset))
        top_sel = top_k(subset, metric, k)
        bot_sel = bottom_k(subset, metric, k)
        rnd_sel = random_k(subset, k, args.seed)

        top_path = out_dir / f"{metric}_top{k}.json"
        bot_path = out_dir / f"{metric}_bottom{k}.json"
        rnd_path = out_dir / f"{metric}_random{k}_seed{args.seed}.json"

        write_json(top_path, top_sel)
        write_json(bot_path, bot_sel)
        write_json(rnd_path, rnd_sel)

        print(f"[{metric}] available={len(subset)}  top={len(top_sel)}  bottom={len(bot_sel)}  random={len(rnd_sel)}")
        print(f"  -> {top_path.name}\n  -> {bot_path.name}\n  -> {rnd_path.name}")

if __name__ == "__main__":
    main()
