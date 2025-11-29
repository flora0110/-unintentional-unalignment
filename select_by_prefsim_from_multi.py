import json
import os

# 路徑自己改成你實際存的檔案
INPUT_JSON_PATH = "/scratch/user/chuanhsin0110/LLMRec-Labs/unintentional-unalignment/outputs/goodreads_beam/item_pref_similarity.json"
OUTPUT_JSON_PATH = "/scratch/user/chuanhsin0110/LLMRec-Labs/unintentional-unalignment/data_files/goodreads/item_pref_similarity_best_reject_by_ches.json"


def main():
    # 讀取之前存好的 json_friendly 結果
    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        all_samples = json.load(f)  # list[dict]

    selected_results = []

    for sample in all_samples:
        ches_scores = sample.get("ches_scores", {})
        if not ches_scores:
            # 如果這個 sample 沒有任何候選（沒有出現在 id2name 的 rejected），就跳過
            continue

        # 1) 根據 CHES 挑出最小的那個 candidate（越小 = 你定義的 "最差" reject）
        #    key 是書名（不帶外層引號），value 是 ches 值
        best_title, best_ches = min(ches_scores.items(), key=lambda kv: kv[1])

        # 2) 從其他 metric dict 裡拿對應的數值（如果有）
        med_dict = sample.get("minus_normalized_edit_distances", {})
        ln_ches_dict = sample.get("ln_ches_scores", {})
        last_inner_dict = sample.get("last_hidden_embedding_inner_prods", {})

        best_med = med_dict.get(best_title, None)
        best_ln_ches = ln_ches_dict.get(best_title, None)
        best_last_inner = last_inner_dict.get(best_title, None)

        # 3) 組成你要的輸出格式
        #    注意：原本你的 chosen/rejected 是帶外層的引號形式，所以這裡也把 title 再包回去
        out_entry = {
            "prompt": sample["prompt"],
            "chosen": sample["chosen"],  # 保留原樣（裡頭已經是 "\"Title\"" 的格式）
            "rejected": f"\"{best_title}\"",
            "ches_score": best_ches,
            "minus_normalized_edit_distance": best_med,
            "ln_ches_score": best_ln_ches,
            "last_hidden_embedding_inner_prod": best_last_inner,
        }

        selected_results.append(out_entry)

    # 存成新的 JSON 檔
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(selected_results, f, ensure_ascii=False, indent=2)

    print(f"Done. Selected {len(selected_results)} samples.")
    print(f"Saved to: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
