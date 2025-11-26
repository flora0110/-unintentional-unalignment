import re
import torch
import matplotlib.pyplot as plt
import os, getpass

from collections import Counter

import os, re, json, unicodedata
import numpy as np
from collections import Counter


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel
from datasets import Dataset

def save_exposure_counter(counter_obj, path):
    """
    將 Counter 存成 JSON（鍵轉字串、值轉整數）。
    """
    if not isinstance(counter_obj, Counter):
        counter_obj = Counter(counter_obj)
    payload = {str(k): int(round(v)) for k, v in counter_obj.items()}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    return path

def load_exposure_counter(path) -> Counter:
    """
    從 JSON 載回 Counter（值自動轉 int）。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Counter({str(k): int(round(v)) for k, v in data.items()})


def safe_write_json(file_path, data, indent=2):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    base_path, ext = os.path.splitext(file_path)
    final_path = file_path
    counter = 1
    while os.path.exists(final_path):
        print(f"Warning: {final_path} already exists. Generating new file name...")
        final_path = f"{base_path}_{counter}{ext}"
        counter += 1

    with open(final_path, "w") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    print(f"Saved data to {final_path}")

def safe_load_json(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)

def prepare_output_dir(output_path: str, check_subdir: str = "final_model", allow_existing: bool = False) -> str:
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        print(f"Created base output dir: {output_path}")

    if allow_existing:
        if check_subdir is None:
            print(f"Using existing output directory: {output_path}")
            return output_path
        else:
            subdir = os.path.join(output_path, check_subdir)
            if not os.path.exists(subdir):
                os.makedirs(subdir, exist_ok=True)
                print(f"Created subdirectory: {subdir}")
            else:
                print(f"Using existing subdirectory: {subdir}")
            return subdir

    if check_subdir is None:
        if len(os.listdir(output_path)) == 0:
            print(f"Warning: Output directory '{output_path}' exists and is empty. No need to create a new directory.")
            return output_path
        else:
            base_path = output_path
            counter = 1
            final_output_path = base_path
            while os.path.exists(final_output_path) and os.listdir(final_output_path):
                print(f"Warning: '{final_output_path}' already exists and is not empty. Generating a new output directory name...")
                final_output_path = f"{base_path}_{counter}"
                counter += 1
            os.makedirs(final_output_path, exist_ok=True)
            print(f"Using output directory: {final_output_path}")
            return final_output_path
    else:
        final_model_dir = os.path.join(output_path, check_subdir) if check_subdir else output_path
        counter = 1
        final_output_path = output_path
        while os.path.exists(final_model_dir):
            print(f"Warning: '{final_model_dir}' already exists. Generating a new output directory name...")
            final_output_path = f"{output_path}_{counter}"
            final_model_dir = os.path.join(final_output_path, check_subdir) if check_subdir else final_output_path
            counter += 1
        os.makedirs(final_output_path, exist_ok=True)
        print(f"Using output directory: {final_output_path}")
        return final_output_path




def normalize_title(s: str) -> str:
    """去頭尾引號、trim、壓縮內部空白、Unicode 正規化。"""
    if not isinstance(s, str):
        return ""
    # Unicode normalize
    s = unicodedata.normalize("NFKC", s)
    # 去頭尾雙引號/單引號
    print(f"s[0]: {s[0]}")
    print(f"s[-1]: {s[-1]}")
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
        s = s[1:-1]
    # 去前後空白
    s = s.strip()
    # 內部多空白壓成單一空白
    s = re.sub(r"\s+", " ", s)
    return s

def parse_titles(text: str):
    if not isinstance(text, str): return []
    raw = re.findall(r'"([^"]+)"', text)
    return [normalize_title(t) for t in raw]

def extract_first_quoted(s: str) -> str:
    # print(f"origin s: {repr(s)}")
    if not isinstance(s, str): return ""
    m = re.search(r'"(.*?)"', s)
    return normalize_title(m.group(1)) if m else s





def format_prompt(instruction, input_text):
    """
    Format a prompt string given an instruction and optional input text.

    Args:
        instruction (str): The instruction text.
        input_text (str): The supplementary input text.

    Returns:
        str: A formatted prompt.
    """
    if input_text.strip():
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:"
    else:
        return f"### Instruction:\n{instruction}\n\n### Response:"

def generate_predictions_batch(model, tokenizer, dataset, batch_size=8, max_new_tokens=50):
    """
    Generate predictions in batch for a given dataset using the provided model and tokenizer.

    Args:
        model: The language model used for generation.
        tokenizer: The corresponding tokenizer.
        dataset: A HuggingFace Dataset object containing samples with 'instruction', 'input', and 'output'.
        batch_size (int, optional): The batch size for generation. Default is 8.
        max_new_tokens (int, optional): The maximum new tokens generated per sample. Default is 50.

    Returns:
        list: A list of dictionaries, each containing:
              - prompt: The formatted prompt.
              - prediction: The generated prediction (model output after the prompt).
              - ground_truth: The expected output from the dataset.
    """
    results = []
    model.eval()
    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating Predictions"):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        prompts = [format_prompt(sample["instruction"], sample["input"]) for sample in batch]

        # Tokenize the batch of prompts
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        for j, output in enumerate(outputs):
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            # Remove the prompt portion to obtain the prediction
            response = decoded[len(prompts[j]):].strip()
            results.append({
                "prompt": prompts[j],
                "prediction": response,
                "ground_truth": batch[j]["output"].strip('" \n')
            })
    return results

def generate_predictions(config):
    base_model = config["base_model"]
    # use_lora = config.get("use_lora", True)
    finetuned_path = config.get("finetuned_path", "")
    test_data_path = config["test_data_path"]
    output_dir = config["output_dir"]
    batch_size = config.get("batch_size", 8)
    max_new_tokens = config.get("max_new_tokens", 50)
    test_sample_size = config.get("test_sample_size", 1000)
    print(f"output_dir: {output_dir}")
    predict_dir = prepare_output_dir(output_dir, "predictions")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", local_files_only=True)
    if finetuned_path:
        model = PeftModel.from_pretrained(model, finetuned_path)
    # If using multiple GPUs, wrap with DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for inference...")
        model = torch.nn.DataParallel(model)
        inference_model = model.module  # use model.module for inference in DataParallel mode
    else:
        inference_model = model

    # Load test dataset and select the required number of samples
    dataset = safe_load_json(test_data_path)
    dataset = Dataset.from_list(dataset)

    # Generate predictions
    results = generate_predictions_batch(inference_model, tokenizer, dataset, batch_size=batch_size, max_new_tokens=max_new_tokens)

    # Save predictions
    raw_results_filename = f"raw_results_{test_sample_size}.json"
    raw_results_path = os.path.join(predict_dir, raw_results_filename)

    safe_write_json(raw_results_path, results)

    print(f"\nInference completed. Results saved to: {raw_results_path}")


# EVALUATION 

import os

import re
import math
import json
import torch
import pandas as pd
from tqdm import tqdm
from collections import Counter

def read_json(json_file: str) -> dict:
    """
    Read a JSON file and return its content.

    Args:
        json_file (str): The path to the JSON file.

    Returns:
        dict: Parsed JSON content.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def batch(list_obj, batch_size=1):
    """
    Yield batches of a list.

    Args:
        list_obj (list): The list to be divided.
        batch_size (int): Number of items per batch.

    Yields:
        list: A sublist containing at most batch_size elements.
    """
    chunk_size = (len(list_obj) - 1) // batch_size + 1
    for i in range(chunk_size):
        yield list_obj[batch_size * i: batch_size * (i + 1)]

def sum_of_first_i_keys(sorted_dic, i):
    """
    Sum the values of the first i keys in a sorted dictionary.

    Args:
        sorted_dic (dict): The dictionary sorted by value.
        i (int): Number of keys to sum.

    Returns:
        float: The sum of the first i values.
    """
    keys = list(sorted_dic.values())[:i]
    return sum(keys)

def gh(test_data, eval_dir: str) -> list:
    """
    Compute and normalize genre distribution based on the input test data.

    Args:
        category (str): The category name (e.g., "Goodreads").
        test_data (list): A list of test data entries.
        eval_dir (str): Directory containing evaluation files for the category.

    Returns:
        list: A list of normalized genre values.
    """
    notin_count = 0
    in_count = 0
    name2genre = read_json(os.path.join(eval_dir,  "name2genre.json"))
    genre_dict = read_json(os.path.join(eval_dir,  "genre_dict.json"))
    for data in tqdm(test_data, desc="Processing category data..."):
        input_text = data['prompt']
        names = re.findall(r'"([^"]+)"', input_text)
        for name in names:
            if name in name2genre:
                in_count += 1
                genres = name2genre[name]
            else:
                notin_count += 1
                continue
            select_genres = []
            for genre in genres:
                if genre in genre_dict:
                    select_genres.append(genre)
            if len(select_genres) > 0:
                for genre in select_genres:
                    genre_dict[genre] += 1 / len(select_genres)
    gh_values = [genre_dict[x] for x in genre_dict]
    gh_normalize = [x / sum(gh_values) for x in gh_values]
    print(f"InCount: {in_count}\nNotinCount: {notin_count}")
    return gh_normalize

def update_csv(dataset_name: str,
               model_name: str,
               sample_method: str,
               metrics_dict: dict,
               csv_file: str):
    """
    Update (or create) a CSV of evaluation metrics, keyed by Dataset, Model, and SampleMethod.
    If a row with the same (Dataset, Model, SampleMethod) exists, its metric columns will be overwritten.
    Otherwise a new row is appended.
    """
    # Make sure directory exists
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    key_cols = ["Dataset", "Strategy", "Composition"]
    metric_cols = list(metrics_dict.keys())

    if not os.path.exists(csv_file):
        # Build empty DataFrame with all needed columns
        df = pd.DataFrame(columns=key_cols + metric_cols)
        # New row
        new_row = {"Dataset": dataset_name,
                   "Strategy": model_name,
                   "Composition": sample_method}
        new_row.update(metrics_dict)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.read_csv(csv_file)
        # Ensure the SampleMethod column exists
        if "SampleMethod" not in df.columns:
            df["SampleMethod"] = None

        # Condition: same dataset, model, and sample_method
        cond = (
            (df["Dataset"] == dataset_name) &
            (df["Strategy"] == model_name) &
            (df["Composition"] == sample_method)
        )

        if not cond.any():
            # Append new row
            new_row = {col: None for col in df.columns}
            new_row.update({
                "Dataset": dataset_name,
                "Strategy": model_name,
                "Composition": sample_method,
                **metrics_dict
            })
            # Add any missing metric columns
            for m in metrics_dict:
                if m not in df.columns:
                    df[m] = None
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            # Update existing row
            for metric, value in metrics_dict.items():
                if metric not in df.columns:
                    df[metric] = None
                df.loc[cond, metric] = value

    # Write back
    df.to_csv(csv_file, index=False)
    print(f"CSV updated: {csv_file}")

def lookup_exposure_or_raise(exposure_count: Counter, topi_name: str) -> float:
    key = extract_first_quoted(topi_name)
    if key not in exposure_count:
        print(
            f"exposure_count 缺少鍵：'{key}'（由 topi_name='{topi_name}' 萃取）。"
            "請確認 exposure_count 的鍵與資料名稱一致。"
        )
        return 0
    return float(exposure_count[key])

def evaluate_metrics(config):
    """
    Main evaluation function to compute metrics from prediction results.
    """
    category = config["category"]
    eval_dir = config["eval_dir"]

    id2name = safe_load_json(os.path.join(eval_dir,"id2name.json"))
    name2id = safe_load_json(os.path.join(eval_dir,"name2id.json"))
    embeddings = torch.load(os.path.join(eval_dir,"embeddings.pt"))
    name2genre = safe_load_json(os.path.join(eval_dir, "name2genre.json"))
    genre_dict = safe_load_json(os.path.join(eval_dir, "genre_dict.json"))


    test_data = safe_load_json(config["predictions_file"])

    # ----------- 這裡開始：改成雲端模型預設 -----------
    from sentence_transformers import SentenceTransformer

    # 若沒有提供本地路徑，就用雲端模型 ID
    model_id = config.get("sbert_model_path") or "sentence-transformers/paraphrase-MiniLM-L3-v2"
    sbert_model = SentenceTransformer(model_id)
    print(f"SentenceTransformer loaded: {model_id}")

    # 裝置自動偵測
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sbert_model = sbert_model.to(device)
    print(f"device: {device}")
    # ---------------------------------------------

    # 將預先計算好的候選嵌入搬到相同裝置（節省拷貝）
    embeddings = (embeddings if isinstance(embeddings, torch.Tensor)
                  else torch.tensor(embeddings))
    embeddings = embeddings.to(device)

    # 取出每條 prediction 的第一個引號內名稱，fallback 用第一行文字
    text = []
    for entry in tqdm(test_data, desc="Extracting prediction names"):
        if len(entry.get("prediction","")) > 0:
            match = re.search(r'"([^"]+)"', entry['prediction'])
            if match:
                name = match.group(1)
                text.append(name)
            else:
                text.append(entry['prediction'].split('\n', 1)[0])
        else:
            text.append("NAN")
            print("Empty prediction!")

    pred_not_in_count = sum(1 for name in text if name not in name2genre)
    pred_not_in_ratio = pred_not_in_count / len(text)
    print(f"Prediction not in name2genre ratio: {pred_not_in_ratio:.4f}")

    # ----- 用 SBERT 編碼預測文字（雲端模型 or 本地），批次化、回傳 numpy 更省記憶體 -----
    predict_embeddings = []
    # 可依環境調整，8~64 都可；過大容易 OOM
    encode_bs = int(config.get("encode_batch_size", 32))
    for batch_text in batch(text, encode_bs):
        vec = sbert_model.encode(
            batch_text,
            batch_size=encode_bs,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=device
        )
        predict_embeddings.append(torch.tensor(vec, device=device))
    predict_embeddings = torch.cat(predict_embeddings, dim=0)
    print("Prediction embeddings size:", tuple(predict_embeddings.size()))
    # --------------------------------------------------------------------------------------

    # 距離&排名
    dist = torch.cdist(predict_embeddings, embeddings, p=2)
    batch_size_ = 1
    num_batches = (dist.size(0) + batch_size_ - 1) // batch_size_
    print(f"Number of batches for ranking: {num_batches}")
    rank_list = []
    for i in tqdm(range(num_batches), desc="Processing Batches"):
        start_idx = i * batch_size_
        end_idx = min((i + 1) * batch_size_, dist.size(0))
        batch_dist = dist[start_idx:end_idx]
        batch_rank = batch_dist.argsort(dim=-1).argsort(dim=-1)
        if device == "cuda":
            torch.cuda.empty_cache()
        rank_list.append(batch_rank)
    rank_list = torch.cat(rank_list, dim=0)
    print(f"Rank list length: {len(rank_list)}")

    topk = int(config["topk"])
    S_ndcg = 0
    S_hr = 0
    diversity_set = set()
    diversity_dic = {}
    total = len(test_data)
    exposure_sum_all = 0.0
    for i in tqdm(range(len(test_data)), desc="Calculating Metrics"):
        rank = rank_list[i]
        target_name = test_data[i]['ground_truth'].strip().strip('"')
        if target_name in name2id:
            target_id = name2id[target_name]
        else:
            continue
        rankId = rank[target_id]
        if rankId < topk:
            S_ndcg += (1 / math.log(rankId + 2))
            S_hr += 1

        current_topk_list = []
        for j in range(topk):
            topi_id = torch.argwhere(rank == j).item()
            topi_name = id2name[str(topi_id)]
            exposure_val = lookup_exposure_or_raise(exposure_count, topi_name)

            exposure_sum_all += exposure_val
            if topi_name in name2genre:

                topi_genre = name2genre[topi_name]
                select_genres = [genre for genre in topi_genre if genre in genre_dict]
                if len(select_genres) > 0:
                    for genre in select_genres:
                        genre_dict[genre] += 1 / len(select_genres)
            diversity_set.add(topi_id)
            diversity_dic[topi_id] = diversity_dic.get(topi_id, 0) + 1

            current_topk_list.append({
                "index": j,
                "topi_id": int(topi_id),
                "topi_name": extract_first_quoted(topi_name),
                "topi_exposure": exposure_val,
            })
        test_data[i]["topK"] = current_topk_list

    NDCG = S_ndcg / len(test_data) / (1 / math.log(2))
    HR = S_hr / len(test_data)
    PopRatio = exposure_sum_all / (total * topk) if (total > 0 and topk > 0) else 0.0
    diversity = len(diversity_set)

    gh_genre = gh(test_data, eval_dir)
    gp_genre = [genre_dict[x] for x in genre_dict]
    gp_genre = [x / sum(gp_genre) for x in gp_genre]
    dis_genre = [gp_genre[i] - gh_genre[i] for i in range(len(gh_genre))]
    DGU_genre = max(dis_genre) - min(dis_genre)
    dis_abs_genre = [abs(x) for x in dis_genre]
    MGU_genre = sum(dis_abs_genre) / len(dis_abs_genre)

    eval_dic = {
        "strategy": config.get("strategy", "Unknown"),
        "composition": config.get("composition", "Unknown"),
        "topK": topk,
        "Dis_genre": dis_abs_genre,
        "NDCG": NDCG,
        "HR": HR,
        "diversity": diversity,
        "DivRatio": diversity / (total * topk),
        "DGU": DGU_genre,
        "MGU": MGU_genre,
        "PopRatio": PopRatio,
        "Predict_NotIn_Ratio": pred_not_in_ratio,

    }
    sorted_dic = dict(sorted(diversity_dic.items(), key=lambda item: item[1], reverse=True))
    eval_dic["ORRatio"] = sum_of_first_i_keys(sorted_dic, 3) / (topk * total)
    print(f"ORRatio: {eval_dic['ORRatio']}")

    output_file = config["output_file"]
    print(f"Output file: {output_file}")
    output_dir = os.path.dirname(output_file)
    print(f"Output directory: {output_dir}")
    prepare_output_dir(output_dir, None, allow_existing=True)
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    data.append(eval_dic)

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, separators=(',', ': '), indent=2)

    topk_file = config["topK_file"]
    topk_dir = os.path.dirname(topk_file)
    prepare_output_dir(topk_dir, None, allow_existing=True)
    # 需求是「格式直接在 test_data 下新增」，所以直接把增添了 "topK" 的 test_data 存出去
    with open(topk_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    if config.get("exp_csv"):
        metric_dic = {
            f"MGU@{topk}": eval_dic["MGU"],
            f"DGU@{topk}": eval_dic["DGU"],
            f"DivRatio@{topk}": eval_dic["DivRatio"],
            f"ORRatio@{topk}": eval_dic["ORRatio"],
            f"PredictNotInRatio@{topk}": eval_dic["Predict_NotIn_Ratio"],
            f"NDCG@{topk}": eval_dic["NDCG"],
            f"HR@{topk}": eval_dic["HR"],
            f"PopRatio@{topk}": eval_dic["PopRatio"],
            "Predict_NotIn_Ratio": eval_dic["Predict_NotIn_Ratio"],
        }
        update_csv(category, config.get("strategy", "Unknown"),
                   config.get("composition", "Unknown"),
                   metric_dic, config["exp_csv"])
    print("Evaluation complete.")


if __name__ == "__main__":
    with open("/scratch/user/chuanhsin0110/LLMRec-Labs/unintentional-unalignment/data_files/goodreads/exposure_count.json", 'r', encoding='utf-8') as f:
        exposure_count =  json.load(f)

    # BASE_MODEL = "Qwen/Qwen3-0.6B"
    distance  = "DPO"
    # distance = "last_hidden_embedding_inner_prod"
    composition = "RN1_epoch1"
    # composition = "bottom100"
    # composition = "random100_seed123"

    PREDICTION_DIR = f"/scratch/user/chuanhsin0110/LLMRec-Labs/unintentional-unalignment/outputs/Goodreads_test/predictions/{distance}_{composition}/"
    METRICS_DIR = f"/scratch/user/chuanhsin0110/LLMRec-Labs/unintentional-unalignment/outputs/Goodreads_test/metrics/{distance}_{composition}/"
    METRICS_DIR_ROOT = f"/scratch/user/chuanhsin0110/LLMRec-Labs/unintentional-unalignment/outputs/Goodreads_test/metrics/"
    CONFIG_PRIDICTION = {
        # Model settings
        "base_model": "/scratch/user/chuanhsin0110/hf_models/allenai-OLMo-1B-hf",
        "finetuned_path": f"/scratch/user/chuanhsin0110/LLMRec-Labs/unintentional-unalignment/outputs/Goodreads_test/{distance}_{composition}/final_model/",


        # Data settings
        "test_data_path": "/scratch/user/chuanhsin0110/LLMRec-Labs/unintentional-unalignment/data_files/goodreads/sampled/test.json",

        # Output settings
        "strategy": distance,
        "composition": composition,
        # min_beta: 0.01
        # max_beta: 0.15
        "num_train_epochs": 1,
        "output_dir": PREDICTION_DIR,
        "temperature": 1.0,
        "beta": 0.11,
        "batch_size": 8,
        "max_new_tokens": 50,
        "test_sample_size": 1000,
    }

    CONFIG_EVAL = {

    # Evaluation configuration for Goodreads

    # Evaluation directory where auxiliary eval files are stored.
    "eval_dir": f"/scratch/user/chuanhsin0110/LLMRec-Labs/unintentional-unalignment/data_files/goodreads/eval",

    # Category (subfolder within eval_dir).
    "category": "Goodreads",

    # Prediction results file (input)
    "predictions_dir": PREDICTION_DIR,
    "predictions_filename": "raw_results_1000.json",
    "predictions_file": f"{PREDICTION_DIR}/raw_results_1000.json",


    # predictions_file: "/scratch/user/chuanhsin0110/ClusterExposure-DPO/experiments/predictions/DS-DPO/Div_10_1.0/dpc_0.08_3.0/raw_results_1000.json"
    # predictions_file: "/scratch/user/chuanhsin0110/test_0321/nlp/predictions/raw_results_1000.json"

    # Output file for evaluation results (JSON)
    # tuned_model: "sigmoid_DS-DPO"
    "strategy": distance,
    "composition": composition,
    "output_file": f"{METRICS_DIR}/eval_result_wpop.json",
    "hmt_lift_file" : f"{METRICS_DIR}/head_mid_tail_lift.json",
    "hmt_lift_pre_test_file" : f"{METRICS_DIR}/head_mid_tail_pre_test_lift.json",
    "topK_file": f"{METRICS_DIR}/topK.json",


    "plot_file": f"{METRICS_DIR}/topK_plot.png",
    "ceil_plot_file": f"{METRICS_DIR}/ceil_topK_plot.png",
    "lift_plot_file": f"{METRICS_DIR}/lift_topK_plot.png",
    "rate_plot_file": f"{METRICS_DIR}/rate_topK_plot.png",
    "exposure_count_file": "/scratch/user/chuanhsin0110/LLMRec-Labs/unintentional-unalignment/data_files/goodreads/exposure_count.json",

    "hmt_accurancy_file": f"{METRICS_DIR}/hmt_accurancy.json",
    # "base_line_topK_file": f"{GDRIVE_ROOT}/experiences/metrics/SFT-tuned/topK.json",

    # min_beta: 0.01
    # max_beta: 0.15
    "beta": 0.11,
    "num_train_epochs": 1,


    # output_file: "/scratch/user/chuanhsin0110/ClusterExposure-DPO/metrics/predictions/DS-DPO/Div_10_1.0/dpc_0.08_3.0/eval_result.json"
    # output_file: "/scratch/user/chuanhsin0110/test_0321/nlp/metrics/eval_result.json"

    # Optional CSV file to update evaluation metrics.
    "exp_csv": f"{METRICS_DIR_ROOT}/metrics_summary_wpop.csv",
    "hmt_acc_csv": f"{METRICS_DIR_ROOT}/metrics_summary_hmt_acc.csv",
    # exp_csv: "/scratch/user/chuanhsin0110/test_0321/nlp/metrics/metrics_summary.csv"
    # Evaluation parameter: top-k for ranking metrics.
    "topk": 5,

    # Model name and sampling method (for record purposes)
    "temperature": 1.0,

    # model_name: "DS-DPO"
    # sample_method: "dpc_0.08_3.0"

    # SentenceTransformer model path for encoding predictions.
    "sbert_model_path": "/scratch/user/chuanhsin0110/ClusterExposure-DPO/models/paraphrase-MiniLM-L3-v2",

    # "raw_train_data": f"{RAW_DATA_DIR}/train.json",
    # "raw_valid_data": f"{RAW_DATA_DIR}/valid.json",

    }

    # generate_predictions(CONFIG_PRIDICTION)
    evaluate_metrics(CONFIG_EVAL)