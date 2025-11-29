import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# ==== 你原本的 normalized edit distance 實作 ==== 

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


# ==== 路徑 & 模型名稱（自己改） ==== 

GOODREADS_PATH = "/scratch/user/chuanhsin0110/LLMRec-Labs/unintentional-unalignment/data_files/goodreads/beam_valid.json"
ID2NAME_PATH = "/scratch/user/chuanhsin0110/LLMRec-Labs/unintentional-unalignment/data_files/goodreads/eval/id2name.json"
MODEL_NAME = "/scratch/user/chuanhsin0110/LLMRec-Labs/unintentional-unalignment/outputs/goodreads_models/olmo1b_sft_goodreads_cross_entropy_2025_11_05-05_08_27/model_epoch_0"

RESULTS_PT_PATH = "/scratch/user/chuanhsin0110/LLMRec-Labs/unintentional-unalignment/outputs/goodreads_beam/item_pref_similarity.pt"
RESULTS_JSON_PATH = "/scratch/user/chuanhsin0110/LLMRec-Labs/unintentional-unalignment/outputs/goodreads_beam/item_pref_similarity.json"


def main():
    # ==== 讀 data & id2name ==== 

    with open(GOODREADS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)   # list[dict]，每個 dict 有 prompt / chosen / rejected

    with open(ID2NAME_PATH, "r", encoding="utf-8") as f:
        id2name = json.load(f)

    valid_titles = set(id2name.values())

    # ==== 載入 model / tokenizer ==== 

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    results = []          # 給 torch.save 用的 python list
    json_friendly = []    # 給 json.dump 用的 list

    for sample_idx, example in tqdm(list(enumerate(data)), desc="Processing samples"):
        prompt = example["prompt"]
        chosen_raw = example["chosen"]
        # 去掉外層的雙引號（你的格式是 "\"Title\""）
        chosen_title = chosen_raw.strip('"')

        minus_norm_ed = {}
        ches_scores = {}
        ln_ches_scores = {}
        last_inner_prods = {}

        # --- loop over rejected candidates --- 
        for rej_raw in example["rejected"]:
            r_title = rej_raw.strip('"')

            # 只保留出現在 id2name 內的項目
            if r_title not in valid_titles:
                continue

            query = prompt
            text_w = query + chosen_title
            text_l = query + r_title

            # ---- tokenize（模仿 __goodreads_create_format_input_func 之後的 tokenize_examples）----
            q_ids = tokenizer(query, padding=False, truncation=False, add_special_tokens=False).input_ids
            w_ids = tokenizer(text_w, padding=False, truncation=False, add_special_tokens=False).input_ids
            l_ids = tokenizer(text_l, padding=False, truncation=False, add_special_tokens=False).input_ids

            query_len = len(q_ids)
            pref_ids = w_ids[query_len:]
            dispref_ids = l_ids[query_len:]

            if len(pref_ids) == 0 or len(dispref_ids) == 0:
                # 如果有奇怪的 truncate 導致沒有回答部分，就跳過
                continue

            # ===== 1) normalized edit distance ===== 
            d = normalized_edit_distance(pref_ids, dispref_ids)
            minus_norm_ed[r_title] = -float(d)

            # ===== 2) CHES / ln-CHES / last hidden inner product ===== 

            pref_tensor = torch.tensor(w_ids, dtype=torch.long, device=device)
            dispref_tensor = torch.tensor(l_ids, dtype=torch.long, device=device)

            with torch.no_grad():
                pref_out = model(input_ids=pref_tensor.unsqueeze(0), output_hidden_states=True)
                dispref_out = model(input_ids=dispref_tensor.unsqueeze(0), output_hidden_states=True)

            hidden_w = pref_out.hidden_states[-1][0]   # [L_w, H]
            hidden_l = dispref_out.hidden_states[-1][0]  # [L_l, H]

            # 和原 script 一樣，從 query_len - 1 開始切
            preferred_hidden_embed = hidden_w[query_len - 1:]     # [T_w, H]
            dispreferred_hidden_embed = hidden_l[query_len - 1:]  # [T_l, H]

            if preferred_hidden_embed.shape[0] == 0 or dispreferred_hidden_embed.shape[0] == 0:
                continue

            S_w = preferred_hidden_embed.sum(dim=0)
            S_l = dispreferred_hidden_embed.sum(dim=0)
            T_w = preferred_hidden_embed.shape[0]
            T_l = dispreferred_hidden_embed.shape[0]

            # CHES
            ches = (S_w * S_l).sum() - torch.norm(S_w) ** 2

            # ln-CHES
            pref_dispref = (S_w * S_l).sum() / (T_w * T_l)
            pref_only = torch.norm(S_w) ** 2 / (T_w ** 2)
            ln_ches = pref_dispref - pref_only

            # last hidden inner product
            last_inner = torch.inner(
                preferred_hidden_embed[-1],
                dispreferred_hidden_embed[-1]
            )

            ches_scores[r_title] = float(ches.detach().cpu())
            ln_ches_scores[r_title] = float(ln_ches.detach().cpu())
            last_inner_prods[r_title] = float(last_inner.detach().cpu())

        # ===== 組 per-sample 結果（含 sample_indices）===== 

        sample_result = {
            "sample_indices": torch.tensor([sample_idx]),   # 跟你要的格式：tensor([i])
            "minus_normalized_edit_distances": minus_norm_ed,
            "ches_scores": ches_scores,
            "ln_ches_scores": ln_ches_scores,
            "last_hidden_embedding_inner_prods": last_inner_prods,
        }
        results.append(sample_result)

        # ===== 組 JSON-friendly 結果（用 prompt / chosen 代替 sample_indices）===== 

        json_friendly.append({
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "minus_normalized_edit_distances": minus_norm_ed,
            "ches_scores": ches_scores,
            "ln_ches_scores": ln_ches_scores,
            "last_hidden_embedding_inner_prods": last_inner_prods,
        })

    # ==== 存起來 ==== 

    # 1) torch 格式（含 tensor）
    torch.save(results, RESULTS_PT_PATH)

    # 2) 純 JSON 格式（不能有 tensor）
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(json_friendly, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved {len(results)} samples.")
    print(f"PT results:   {RESULTS_PT_PATH}")
    print(f"JSON results: {RESULTS_JSON_PATH}")


if __name__ == "__main__":
    main()
