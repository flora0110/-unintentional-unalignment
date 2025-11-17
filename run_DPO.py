import os, json

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





import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, TrainerCallback
from peft import PeftModel, prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from accelerate import Accelerator
import json


def load_model(base_model_name: str, resume_from_checkpoint: str):
    """
    回傳：(policy_model, reference_model, tokenizer)

    - policy_model: base + (可選) LoRA
    - reference_model: 永遠是 *純 base model*，不載任何 LoRA，且凍結
    """
    accelerator = Accelerator()

    # 你有需要再開 4-bit；這裡先以 FP16/bfloat16 自動映射為主
    # from transformers import BitsAndBytesConfig
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True, bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )

    # -------- Policy model --------
    policy_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        # quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )
    policy_model.config.use_cache = False
    # 若你真的用 k-bit，才需要這行；否則可註解
    # policy_model = prepare_model_for_kbit_training(policy_model)

    if resume_from_checkpoint != "base_model":
        # 載入既有的 LoRA/PEFT 權重作為 policy
        policy_model = PeftModel.from_pretrained(
            policy_model,
            resume_from_checkpoint,
            is_trainable=True
        )
    else:
        # 你原本這段有變數名錯誤，修正如下：建立全新 LoRA 頭、讓 policy 可訓練
        peft_config = LoraConfig(
            inference_mode=False,
            r=16,
            lora_alpha=32,
            target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        policy_model = get_peft_model(policy_model, peft_config)

    # 顯示可訓練參數（確認只有 LoRA 在訓練）
    try:
        policy_model.print_trainable_parameters()
    except Exception:
        pass

    # -------- Reference model（= 乾淨 base，凍結）--------
    reference_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        # quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )
    # 若你真的用 k-bit，才需要這行；否則可註解
    # reference_model = prepare_model_for_kbit_training(reference_model)

    # 不載入任何 LoRA/PEFT 權重！直接把參數凍結
    reference_model.requires_grad_(False)
    reference_model.eval()
    # reference 不需要關閉 cache；保持預設即可

    # -------- Tokenizer --------
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    return policy_model, reference_model, tokenizer


def train_dpo(config):
    """
    Train a model using Direct Preference Optimization (DPO).

    The function:
      - Loads the policy and reference models along with the tokenizer.
      - Loads the training and evaluation datasets from JSON lines files.
      - Prepares training arguments from the configuration.
      - Initializes and runs the DPOTrainer.
      - Saves the best model to the specified output directory.

    Args:
        config (dict): Configuration dictionary with parameters for model,
                       data paths, and DPO training settings.

    Returns:
        None
    """
    base_model_name = config["base_model"]
    resume_from_checkpoint = config["resume_from_checkpoint"]
    output_dir = config["output_dir"]

    output_dir = prepare_output_dir(output_dir)

    # Load models and tokenizer
    policy_model, reference_model, tokenizer = load_model(base_model_name, resume_from_checkpoint)

    # Load DPO training dataset (JSON lines format: one JSON per line)
    train_data = safe_load_json(config["train_data_path"])
    # valid_data = safe_load_json(config["valid_data_path"])

    # Convert lists to HuggingFace Dataset objects
    train_dataset = Dataset.from_list(train_data)
    # eval_dataset = Dataset.from_list(valid_data)

    # Prepare training arguments from the dpo section of config
    training_args = DPOConfig(
        beta = config["dpo"]["beta"],
        per_device_train_batch_size = config["dpo"]["per_device_train_batch_size"],
        per_device_eval_batch_size = config["dpo"]["per_device_eval_batch_size"],
        gradient_accumulation_steps = config["dpo"]["gradient_accumulation_steps"],
        warmup_steps = config["dpo"]["warmup_steps"],
        num_train_epochs = config["dpo"]["num_train_epochs"],
        learning_rate = float(config["dpo"]["learning_rate"]),
        bf16 = config["dpo"]["bf16"],
        logging_steps = config["dpo"]["logging_steps"],
        optim = config["dpo"]["optim"],
        # eval_strategy = config["dpo"]["evaluation_strategy"],
        save_strategy = config["dpo"]["save_strategy"],
        output_dir = output_dir,
        save_total_limit = config["dpo"].get("save_total_limit", 1),
        load_best_model_at_end = config["dpo"].get("load_best_model_at_end", True),
        max_prompt_length = config["dpo"].get("max_prompt_length", 512),
        max_length = config["dpo"].get("max_length", 512),
        report_to="none",
    )

    # Initialize the DPOTrainer with callbacks for early stopping and loss threshold
    trainer = DPOTrainer(
        model = policy_model,
        ref_model = reference_model,
        args = training_args,
        train_dataset = train_dataset,
        # eval_dataset = eval_dataset,
        processing_class = tokenizer,
    )

    # Start the training process
    trainer.train()
    trainer.save_state()

    # Save the trained model and tokenizer in the output directory
    final_output_dir = os.path.join(output_dir, "final_model")
    trainer.model.save_pretrained(final_output_dir, safe_serialization=False)
    tokenizer.save_pretrained(final_output_dir)
    print("\nTraining completed. Best model saved to:", final_output_dir)


if __name__ == "__main__":
    print("Starting DPO training...")

    # distance  = "ches"
    distance = "last_hidden_embedding_inner_prod"
    composition = "top100"

    CONFIG_DPO = {

        #"distance": "ches",
        # "distance": "ln_ches",
        "distance": distance,
        "composition": composition,

        # Model and checkpoint
        "base_model": "/scratch/user/chuanhsin0110/hf_models/allenai-OLMo-1B-hf",
        # "resume_from_checkpoint":  "/scratch/user/chuanhsin0110/LLMRec-Labs/unintentional-unalignment/outputs/goodreads_models/olmo1b_sft_goodreads_cross_entropy_2025_11_05-05_08_27/model_epoch_0",
        "resume_from_checkpoint": "base_model",
        # Dataset paths
        "train_data_path": f"/scratch/user/chuanhsin0110/LLMRec-Labs/unintentional-unalignment/data_files/goodreads/subsets/{distance}_{composition}.json",
        # "valid_data_path": f"{DATA_INPUT_DIR}/{strategy}/{composition}/valid.json",

        # Output
        "output_dir": f"/scratch/user/chuanhsin0110/LLMRec-Labs/unintentional-unalignment/outputs/Goodreads_test/{distance}_{composition}/",

        # DPO training parameters
        "dpo": {
        "beta": 0.1,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 20,
        "num_train_epochs": 10,
        "learning_rate": 1e-7,
        "bf16": True,
        "logging_steps": 1,
        "optim": "adamw_torch",
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "load_best_model_at_end": True,
        "max_prompt_length": 512,
        "max_length": 512,
        "early_stopping_patience": 2,
        "loss_threshold": 0.05,
        }

    }
    print(f"{CONFIG_DPO['distance']}_score_{CONFIG_DPO['composition']}")
    train_dpo(CONFIG_DPO)