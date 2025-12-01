import json
import os
import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# ==== 1) 你原本的 normalized edit distance 實作 ==== 

def normalized_edit_distance(seq1, seq2):
    len_sent2 = len(seq2)
    dold = list(range(len_sent2 + 1))
    dnew = [0 for _ in range(len_sent2 + 1)]

    for i in range(1, len(seq1) + 1):
        dnew[0] = i
        for j in range(1, len_sent2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dnew[j] = dold[j - 1]
            else:
                substitution = dold[j - 1] + 1
                insertion = dnew[j - 1] + 1
                deletion = dold[j] + 1
                dnew[j] = min(substitution, insertion, deletion)
        dnew, dold = dold, dnew

    return int(dold[-1]) / max(len(seq1), len(seq2))


# ==== 2) 路徑 & 模型設定 ==== 

GOODREADS_PATH = "/scratch/user/chuanhsin0110/LLMRec-Labs/unintentional-unalignment/data_files/goodreads/train.json"
ID2NAME_PATH = "/scratch/user/chuanhsin0110/LLMRec-Labs/unintentional-unalignment/data_files/goodreads/eval/id2name.json"
MODEL_NAME = "/scratch/user/chuanhsin0110/LLMRec-Labs/unintentional-unalignment/outputs/goodreads_models/olmo1b_sft_goodreads_cross_entropy_2025_11_05-05_08_27/model_epoch_0"

# 存檔 prefix，會變成 *_chunk000.pt / *_chunk000.json
OUTPUT_PREFIX_PT = "/scratch/user/chuanhsin0110/LLMRec-Labs/unintentional-unalignment/data_files/goodreads/train/random50_item_pref_similarity"
OUTPUT_PREFIX_JSON = "/scratch/user/chuanhsin0110/LLMRec-Labs/unintentional-unalignment/data_files/goodreads/train/random50_item_pref_similarity"

NUM_RANDOM_ITEMS = 50
RANDOM_SEED = 42

CHUNK_SIZE = 100          # 每 100 筆有效 sample 存一個檔
# MAX_EFFECTIVE_SAMPLES = 1024  # 總共要 1024 筆有效資料


def save_chunk(chunk_idx, results_chunk, json_chunk):
    """
    把目前這一批 chunk 存成一個 pt 檔和一個 json 檔。
    chunk_idx 從 0 開始。
    """
    if not results_chunk:
        return

    pt_path = f"{OUTPUT_PREFIX_PT}_chunk{chunk_idx:03d}.pt"
    json_path = f"{OUTPUT_PREFIX_JSON}_chunk{chunk_idx:03d}.json"

    os.makedirs(os.path.dirname(pt_path), exist_ok=True)

    torch.save(results_chunk, pt_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_chunk, f, ensure_ascii=False, indent=2)

    print(f"[Chunk {chunk_idx}] Saved {len(results_chunk)} samples.")
    print(f"  PT   -> {pt_path}")
    print(f"  JSON -> {json_path}")


def main():
    random.seed(RANDOM_SEED)

    # ==== 3) 讀 GoodReads data & id2name ==== 

    with open(GOODREADS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(ID2NAME_PATH, "r", encoding="utf-8") as f:
        id2name = json.load(f)

    all_titles = list(id2name.values())  # 全部 item pool


    max_effective_samples = len(data)

    # ==== 4) 載入 model / tokenizer ==== 

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # 用來存「目前這一批 chunk」的結果
    results_chunk = []
    json_chunk = []
    chunk_idx = 0

    effective_count = 0  # 累積的有效 sample 數量

    for sample_idx, example in tqdm(list(enumerate(data)), desc="Processing samples"):
        if effective_count >= max_effective_samples:
            break  # 已經有 1024 筆有效資料就停

        prompt = example["prompt"]
        chosen_raw = example["chosen"]
        chosen_title = chosen_raw.strip('"')

        # ==== 5) 從全體 item 中隨機抽 NUM_RANDOM_ITEMS 本（排除 chosen 自己）==== 

        candidate_pool = [t for t in all_titles if t != chosen_title]
        if not candidate_pool:
            continue

        k = min(NUM_RANDOM_ITEMS, len(candidate_pool))
        sampled_titles = random.sample(candidate_pool, k=k)

        minus_norm_ed = {}
        ches_scores = {}
        ln_ches_scores = {}
        last_inner_prods = {}

        for r_title in sampled_titles:
            query = prompt
            text_w = query + chosen_title
            text_l = query + r_title

            # ---- tokenize ---- 
            q_ids = tokenizer(query, padding=False, truncation=False, add_special_tokens=False).input_ids
            w_ids = tokenizer(text_w, padding=False, truncation=False, add_special_tokens=False).input_ids
            l_ids = tokenizer(text_l, padding=False, truncation=False, add_special_tokens=False).input_ids

            query_len = len(q_ids)
            pref_ids = w_ids[query_len:]
            dispref_ids = l_ids[query_len:]

            if len(pref_ids) == 0 or len(dispref_ids) == 0:
                continue

            # ==== 7) normalized edit distance ==== 
            d = normalized_edit_distance(pref_ids, dispref_ids)
            minus_norm_ed[r_title] = -float(d)

            # ==== 8) CHES / ln-CHES / last hidden inner product ==== 

            pref_tensor = torch.tensor(w_ids, dtype=torch.long, device=device)
            dispref_tensor = torch.tensor(l_ids, dtype=torch.long, device=device)

            with torch.no_grad():
                pref_out = model(input_ids=pref_tensor.unsqueeze(0), output_hidden_states=True)
                dispref_out = model(input_ids=dispref_tensor.unsqueeze(0), output_hidden_states=True)

            hidden_w = pref_out.hidden_states[-1][0]   # [L_w, H]
            hidden_l = dispref_out.hidden_states[-1][0]  # [L_l, H]

            preferred_hidden_embed = hidden_w[query_len - 1:]
            dispreferred_hidden_embed = hidden_l[query_len - 1:]

            if preferred_hidden_embed.shape[0] == 0 or dispreferred_hidden_embed.shape[0] == 0:
                continue

            S_w = preferred_hidden_embed.sum(dim=0)
            S_l = dispreferred_hidden_embed.sum(dim=0)
            T_w = preferred_hidden_embed.shape[0]
            T_l = dispreferred_hidden_embed.shape[0]

            ches = (S_w * S_l).sum() - torch.norm(S_w) ** 2

            pref_dispref = (S_w * S_l).sum() / (T_w * T_l)
            pref_only = torch.norm(S_w) ** 2 / (T_w ** 2)
            ln_ches = pref_dispref - pref_only

            last_inner = torch.inner(
                preferred_hidden_embed[-1],
                dispreferred_hidden_embed[-1]
            )

            ches_scores[r_title] = float(ches.detach().cpu())
            ln_ches_scores[r_title] = float(ln_ches.detach().cpu())
            last_inner_prods[r_title] = float(last_inner.detach().cpu())

        # 這個 sample 要求「至少有一個 candidate 算出數值」才算有效
        if len(ches_scores) == 0:
            continue

        # ==== 有效 sample，加入目前 chunk ==== 

        sample_result = {
            "sample_indices": torch.tensor([sample_idx]),
            "minus_normalized_edit_distances": minus_norm_ed,
            "ches_scores": ches_scores,
            "ln_ches_scores": ln_ches_scores,
            "last_hidden_embedding_inner_prods": last_inner_prods,
        }
        results_chunk.append(sample_result)

        json_chunk.append({
            "prompt": prompt,
            "chosen": example["chosen"],
            "random_candidates": list(ches_scores.keys()),
            "minus_normalized_edit_distances": minus_norm_ed,
            "ches_scores": ches_scores,
            "ln_ches_scores": ln_ches_scores,
            "last_hidden_embedding_inner_prods": last_inner_prods,
        })

        effective_count += 1

        # 每累積 CHUNK_SIZE 個有效 sample 就存一次
        if len(results_chunk) >= CHUNK_SIZE:
            save_chunk(chunk_idx, results_chunk, json_chunk)
            chunk_idx += 1
            results_chunk = []
            json_chunk = []

        # 如果剛好達到 MAX_EFFECTIVE_SAMPLES，也可以在這裡再檢查一次，安全起見
        if effective_count >= max_effective_samples:
            break

    # 迴圈結束後，若還有殘餘未滿 100 的 chunk，一樣要存
    if results_chunk:
        save_chunk(chunk_idx, results_chunk, json_chunk)

    print(f"Total effective samples: {effective_count}")


if __name__ == "__main__":
    main()
