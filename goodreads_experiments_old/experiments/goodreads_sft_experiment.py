# unintentional-unalignment/goodreads_experiments/experiments/goodreads_sft_experiment.py
from __future__ import annotations
import logging
import os
from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

from common.data.modules import DataModule
from common.evaluation.evaluators import TrainEvaluator, Evaluator, VoidEvaluator, TrainBatchOutputEvaluator
from common.experiment import FitExperimentBase, ExperimentResult
from common.experiment.fit_experiment_base import ScoreInfo
from common.train.callbacks import Callback
from common.train.fit_output import FitOutput

from goodreads_experiments.data.goodreads_sft_datamodule import GoodreadsSFTDataModule
from goodreads_experiments.train.goodreads_sft_trainer import GoodreadsSFTTrainer
from goodreads_experiments.train.token_logits_and_probs_tracker_callback import TokenLogitsAndProbsTrackerCallback

def add_lora(model, lora_params):
    print("\nApplying LoRA")
    cfg = LoraConfig(
        r=lora_params.get("r", 16),
        lora_alpha=lora_params.get("lora_alpha", 32),
        target_modules=lora_params.get("target_modules", "all-linear"),  # 比 q/v_proj 更保險
        lora_dropout=lora_params.get("lora_dropout", 0.0),
        bias=lora_params.get("bias", "none"),
        task_type=lora_params.get("task_type", "CAUSAL_LM"),
        # 若你想用 RS-LoRA，可加：use_rslora=lora_params.get("use_rslora", True),
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model

class GoodreadsSFTExperiment(FitExperimentBase):

    @staticmethod
    def add_experiment_specific_args(parser):
        FitExperimentBase.add_experiment_base_specific_args(parser)

        # dataset / loader
        parser.add_argument("--dataset", type=str, default="data_files/goodreads/train.json")
        parser.add_argument("--num_train_samples", type=int, default=-1)
        parser.add_argument("--train_samples_random_seed", type=int, default=-1)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--max_length", type=int, default=2048)
        parser.add_argument("--load_dataset_to_gpu", action="store_true")

        # model
        parser.add_argument("--model", type=str, default="allenai/OLMo-1B-hf")
        parser.add_argument("--model_cache_dir", type=str, default=None)
        parser.add_argument("--load_model_checkpoint", type=str, default=None)
        parser.add_argument("--is_lora_checkpoint", action="store_true")
        parser.add_argument("--use_lora", action="store_true")
        parser.add_argument("--lora_rank", type=int, default=8)

        # optimization
        parser.add_argument("--optimizer", type=str, default="rmsprop")
        parser.add_argument("--lr", type=float, default=1e-7)
        parser.add_argument("--gradient_accumulation", type=int, default=-1)

        # misc
        parser.add_argument("--save_model", action="store_true")
        parser.add_argument("--save_finegrained_token_metrics", action="store_true")  # not used for SFT but kept for parity

        # the plan still provides kl_coeff/objective; we ignore kl in SFT and accept "cross_entropy"
        parser.add_argument("--objective", type=str, default="cross_entropy")
        parser.add_argument("--kl_coeff", type=float, default=0.0)

    # def create_datamodule(self, config: dict, state: dict, logger: logging.Logger) -> DataModule:
    #     load_dataset_to_device = state["device"] if config["load_dataset_to_gpu"] and len(state["gpu_ids"]) > 0 else None

    #     tokenizer = AutoTokenizer.from_pretrained(
    #         config["model"], trust_remote_code=True, cache_dir=config["model_cache_dir"], local_files_only=True
    #     )

    #     # load base or checkpoint(+merge LoRA)
    #     if not config["is_lora_checkpoint"] or not config["load_model_checkpoint"]:
    #         model_path = config["load_model_checkpoint"] if config["load_model_checkpoint"] else config["model"]
    #         model = AutoModelForCausalLM.from_pretrained(
    #             model_path, device_map=state["device"], trust_remote_code=True,
    #             cache_dir=config["model_cache_dir"], local_files_only=True
    #         )
    #     else:
    #         model = AutoModelForCausalLM.from_pretrained(
    #             config["model"], device_map=state["device"], trust_remote_code=True,
    #             cache_dir=config["model_cache_dir"], local_files_only=True
    #         )
    #         model = PeftModel.from_pretrained(model=model, model_id=config["load_model_checkpoint"], local_files_only=True)
    #         model = model.merge_and_unload()

    #     if config["use_lora"]:
    #         lora_cfg = LoraConfig(
    #             r=config["lora_rank"],
    #             lora_alpha=config["lora_rank"] * 2,
    #             bias="none",
    #             target_modules="all-linear",
    #             use_rslora=True,
    #             task_type="CAUSAL_LM",
    #         )
    #         model = get_peft_model(model, lora_cfg)

    #     # right/right padding; pad_token fallback
    #     tokenizer.padding_side = "right"
    #     tokenizer.truncation_side = "right"
    #     if tokenizer.pad_token is None:
    #         tokenizer.pad_token = tokenizer.eos_token

    #     state["tokenizer"] = tokenizer
    #     state["model"] = model

    #     dm = GoodreadsSFTDataModule(
    #         path=config["dataset"],
    #         tokenizer=tokenizer,
    #         max_length=config["max_length"] if "max_length" in config else 2048,
    #         batch_size=config["batch_size"],
    #         device=state["device"],
    #         load_dataset_to_device=load_dataset_to_device,
    #         num_train_samples=config["num_train_samples"],
    #         random_seed=config["train_samples_random_seed"],
    #     )
    #     dm.setup()
    #     return dm



    def create_datamodule(self, config: dict, state: dict, logger: logging.Logger) -> DataModule:
        # 建議：除非很小的 toy data，別把整個 dataset 搬到 GPU
        load_dataset_to_device = state["device"] if config["load_dataset_to_gpu"] and len(state["gpu_ids"]) > 0 else None

        tokenizer = AutoTokenizer.from_pretrained(
            config["model"],
            trust_remote_code=True,
            cache_dir=config["model_cache_dir"],
            local_files_only=True,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"

        # 4-bit 量化設定
        use_4bit = bool(config.get("use_4bit", False))
        quant_cfg = None
        if use_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
            )

        # 載入 base / checkpoint
        if (not config["is_lora_checkpoint"]) or (not config["load_model_checkpoint"]):
            model_path = config["load_model_checkpoint"] if config["load_model_checkpoint"] else config["model"]

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                cache_dir=config["model_cache_dir"],
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=quant_cfg,
            )

            # 訓練時關掉 cache；若是 QLoRA，先做 k-bit 準備
            model.config.use_cache = False
            if use_4bit and config.get("use_lora", False):
                model = prepare_model_for_kbit_training(model)
                model.gradient_checkpointing_enable()

            # ✅ 這裡用 add_lora()（取代手寫 LoraConfig 區塊）
            if config.get("use_lora", False):
                model = add_lora(model, {
                    "r": config["lora_rank"],
                    "lora_alpha": config["lora_rank"] * 2,
                    "lora_dropout": 0.05,
                    "target_modules": "all-linear",  # 保險；若已確認 OLMo 命名可改成具體列表
                    "task_type": "CAUSAL_LM",
                    # "use_rslora": True,  # 若你的 add_lora 支援，可打開
                })

        else:
            # merge LoRA：在非量化模型上 merge
            model = AutoModelForCausalLM.from_pretrained(
                config["model"],
                trust_remote_code=True,
                cache_dir=config["model_cache_dir"],
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            peft_model = PeftModel.from_pretrained(
                model=model,
                model_id=config["load_model_checkpoint"],
                local_files_only=True
            )
            model = peft_model.merge_and_unload()
            model.config.use_cache = False  # 若後續還要再訓練（非 4-bit）一樣關掉

        state["tokenizer"] = tokenizer
        state["model"] = model

        dm = GoodreadsSFTDataModule(
            path=config["dataset"],
            tokenizer=tokenizer,
            max_length=config.get("max_length", 1024),
            batch_size=config["batch_size"],
            device=state["device"],
            load_dataset_to_device=load_dataset_to_device,
            num_train_samples=config["num_train_samples"],
            random_seed=config["train_samples_random_seed"],
        )
        dm.setup()
        return dm


    def create_model(self, datamodule: DataModule, config: dict, state: dict, logger: logging.Logger) -> nn.Module:
        return state["model"]

    def create_train_and_validation_evaluators(
        self, model: nn.Module, datamodule: DataModule, device, config: dict, state: dict, logger: logging.Logger
    ) -> Tuple[TrainEvaluator, Evaluator]:
        # We only track "train loss" for compatibility with your plan.
        train_evaluator = TrainBatchOutputEvaluator(metric_names=["train loss"], metric_tags=["train loss"])
        return train_evaluator, VoidEvaluator()

    def create_trainer(
        self, model: nn.Module, datamodule: DataModule, train_evaluator: TrainEvaluator, val_evaluator: Evaluator,
        callback: Callback, device, config: dict, state: dict, logger: logging.Logger
    ) -> Trainer:
        opt_name = config["optimizer"].lower()
        if opt_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        elif opt_name == "rmsprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=config["lr"])
        elif opt_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        return GoodreadsSFTTrainer(
            model=model,
            tokenizer=state["tokenizer"],
            optimizer=optimizer,
            train_evaluator=train_evaluator,
            val_evaluator=val_evaluator,
            callback=callback,
            device=device,
            grad_accum_steps=config["gradient_accumulation"] if "gradient_accumulation" in config else -1,
        )

    def get_default_score_info(self, config: dict, state: dict) -> ScoreInfo:
        return ScoreInfo(metric_name="train loss", is_train_metric=True, largest=False, return_best_score=False)

    def on_experiment_end(
        self, model: nn.Module, datamodule: GoodreadsSFTDataModule, trainer: Trainer, fit_output: FitOutput,
        experiment_result: ExperimentResult, config: dict, state: dict, logger: logging.Logger
    ):
        super().on_experiment_end(model, datamodule, trainer, fit_output, experiment_result, config, state, logger)

        if config["save_model"]:
            experiment_dir = state["experiment_dir"]
            save_dir = os.path.join(experiment_dir, f"model_epoch_{trainer.epoch}")
            model.save_pretrained(save_dir)
            state["tokenizer"].save_pretrained(save_dir)

        # log a few raw samples for provenance
        try:
            first = datamodule.dataset.select(torch.arange(min(len(datamodule.dataset), 5)))
            experiment_result.summary["num_train_samples"] = len(datamodule.dataset)
            experiment_result.summary["data"] = {
                "prompt": [ex["prompt"] for ex in first],
                "chosen": [ex["chosen"] for ex in first],
            }
        except Exception:
            pass
