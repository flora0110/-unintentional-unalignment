import json
import random
from pathlib import Path
from typing import List, Dict

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer



CONFIG_DPO = {
    "train_data_path": "/scratch/user/chuanhsin0110/LLMRec-Labs/unintentional-unalignment/data_files/goodreads/train.json",
    "model": "/scratch/user/chuanhsin0110/hf_models/allenai-OLMo-1B-hf",
    "model_cache_dir": "/scratch/user/chuanhsin0110/hf_cache",
    "load_model_checkpoint": "/scratch/user/chuanhsin0110/LLMRec-Labs/unintentional-unalignment/outputs/persona_models/olmo1b_sft_persona_cross_entropy_2025_11_03-18_37_18/model_epoch_0",
    # "seed": 42,
    "sample_n": 10,
    "max_prompt_length": 512,
    "max_length": 512,
    "beta": 0.1,         # DPO 溫度；後續可 sweep
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,  # 2*8 ≈ 16 的有效 batch
    "learning_rate": 5e-6,
    "weight_decay": 0.0,
    "warmup_ratio": 0.03,
    "log_steps": 10,
    "output_dir": "/scratch/user/chuanhsin0110/LLMRec-Labs/outputs/dpo_min_skeleton",
}
