import json
import os

# 輸入 chunk 路徑 pattern（train 底下）
CHUNK_DIR = "/scratch/user/chuanhsin0110/LLMRec-Labs/unintentional-unalignment/data_files/goodreads/train"
CHUNK_PREFIX = "random50_item_pref_similarity_chunk"
CHUNK_START = 0
CHUNK_END = 10  # 000 ~ 010 (含 10)

# 輸出檔案
OUTPUT_JSON_PATH = os.path.join(
    CHUNK_DIR,
    "random50_item_pref_similarity_best_reject_by_ches.json"
)


def main():
    selected_samples = []

    for idx in range(CHUNK_START, CHUNK_END + 1):
        chunk_path = os.path.join(CHUNK_DIR, f"{CHUNK_PREFIX}{idx:03d}.json")
        if not os.path.exists(chunk_path):
            print(f"[Warn] Chunk file not found, skip: {chunk_path}")
            continue

        print(f"Loading {chunk_path} ...")
        with open(chunk_path, "r", encoding="utf-8") as f:
            chunk_data = json.load(f)  # list[dict]

        for sample in chunk_data:
            ches_scores = sample.get("ches_scores", {})
            if not ches_scores:
                # 這個 sample 沒有任何 candidate 的 CHES，就略過
                continue

            # 根據 CHES 值挑最小的那本書
            best_title, best_ches = min(ches_scores.items(), key=lambda kv: kv[1])

            # 組合輸出格式
            # 注意：原本 chosen 是 "\"Title\"" 格式，這裡維持原樣
            # rejected 則把 title 再包一層雙引號，變成 "\"Title\""
            out_entry = {
                "prompt": sample["prompt"],
                "chosen": sample["chosen"],
                "rejected": f"\"{best_title}\"",
                "ches_score": best_ches,
            }

            selected_samples.append(out_entry)

    # 存成單一 JSON
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(selected_samples, f, ensure_ascii=False, indent=2)

    print(f"Done. Total selected samples: {len(selected_samples)}")
    print(f"Saved to: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
