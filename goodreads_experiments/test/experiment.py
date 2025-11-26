import json
import random
from pathlib import Path
from typing import List, Dict

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer
import os

CONFIG_DPO = {
    "train_data_path": "/scratch/user/chuanhsin0110/LLMRec-Labs/unintentional-unalignment/data_files/goodreads/train.json",
    "model": "/scratch/user/chuanhsin0110/hf_models/allenai-OLMo-1B-hf",
    "model_cache_dir": "/scratch/user/chuanhsin0110/hf_cache",
    "load_model_checkpoint": "/scratch/user/chuanhsin0110/LLMRec-Labs/unintentional-unalignment/outputs/persona_models/olmo1b_sft_persona_cross_entropy_2025_11_03-18_37_18/model_epoch_0",
    # "seed": 42,
    "sample_n": 10,
    "max_prompt_length": 512,
    "max_length": 512,
    "beta": 0.1,         # kl_coeff
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,  # 2*8 ≈ 16 的有效 batch
    "learning_rate": 1e-7,
    # "weight_decay": 0.0,
    # "warmup_ratio": 0.03,
    "log_steps": 10,
    "epochs": 101,
    "output_dir": "/scratch/user/chuanhsin0110/LLMRec-Labs/unintentional-unalignment/output/goodreads_test/",
}



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


def main(cfg=CONFIG_DPO):
    all_data = load_json_list(cfg["train_data_path"])
    sampled =all_data[:cfg["sample_n"]]


    for sample in sampled:
        print("Sampled data:", sample)