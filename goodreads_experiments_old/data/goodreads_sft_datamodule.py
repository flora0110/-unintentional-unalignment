# unintentional-unalignment/goodreads_experiments/data/goodreads_sft_datamodule.py
from __future__ import annotations
from typing import Optional, Dict, Any
import datasets
import torch
import torch.utils.data as torch_data
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from common.data.modules import DataModule

class GoodreadsSFTDataModule(DataModule):
    """
    Load a JSON list of {prompt, chosen} for sequence-level SFT.
    Produces: input_ids, attention_mask, labels (prompt masked to -100).
    """
    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
        batch_size: int = 32,
        pin_memory: bool = False,
        device: torch.device = torch.device("cpu"),
        load_dataset_to_device: Optional[torch.device] = None,
        num_train_samples: int = -1,
        random_seed: int = -1,
    ):
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.device = device
        self.load_dataset_to_device = load_dataset_to_device
        self.num_train_samples = num_train_samples
        self.random_seed = random_seed

        # tokenizer defaults (right/right; pad_token fallback to eos)
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.dataset = None
        self.tokenized_dataset = None

    def setup(self):
        # raw json list: [{"prompt": "...", "chosen": "..."}, ...]
        self.dataset = datasets.load_dataset("json", data_files=self.path, split="train")

        # optional sub-sampling with seed
        if self.num_train_samples is not None and self.num_train_samples > 0 and self.num_train_samples < len(self.dataset):
            g = torch.Generator()
            if self.random_seed and self.random_seed > 0:
                g.manual_seed(self.random_seed)
            perm = torch.randperm(len(self.dataset), generator=g)
            self.dataset = self.dataset.select(perm[: self.num_train_samples])

        # map → build {text, prompt_len}
        self.dataset = self.dataset.map(self._build_text_and_prompt_len, num_proc=1)
        # tokenize + label-mask
        self.tokenized_dataset = self.dataset.map(self._tokenize_and_make_labels, batched=True, remove_columns=self.dataset.column_names)
        
        # HF format to torch tensors
        self.tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        # (optional) move to GPU memory for very small datasets (use carefully)
        if self.load_dataset_to_device is not None:
            dev = self.load_dataset_to_device
            self.tokenized_dataset = self.tokenized_dataset.map(
                lambda batch: {k: v.to(dev) for k, v in batch.items()}, batched=True
            )

    def _build_text_and_prompt_len(self, example: Dict[str, Any]) -> Dict[str, Any]:
        prompt = example["prompt"]
        chosen = example["chosen"]

        # If your tokenizer has a chat_template and you prefer it, you could wrap here.
        # For now we keep your "### Instruction / ### Input / ### Response" style already in prompt.
        # We make sure there is exactly one "### Response:" marker at the end of prompt.
        # Your prompt already ends with "### Response:", so we just concatenate chosen.
        text = f"{prompt}{chosen}"
        # prompt_len = len(tokenizer(prompt)["input_ids"]) -> we compute in token space later
        return {"prompt": prompt, "chosen": chosen, "text": text}

    def _tokenize_and_make_labels(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        texts = batch["text"]
        prompts = batch["prompt"]

        tokenized_all = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )

        # compute prompt token lengths to build label masking
        prompt_tok = self.tokenizer(
            prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )

        input_ids = tokenized_all["input_ids"]
        attention_mask = tokenized_all["attention_mask"]

        labels = []
        for ids, p_ids in zip(input_ids, prompt_tok["input_ids"]):
            # mask prompt part
            L = len(ids)
            Lp = min(len(p_ids), L)  # in case truncation changed length
            lbl = [-100] * L
            # only response region gets labels (next-token prediction standard CE)
            for i in range(Lp, L):
                lbl[i] = ids[i]
            labels.append(lbl)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def train_dataloader(self) -> torch_data.DataLoader:
        bs = self.batch_size if self.batch_size > 0 else len(self.tokenized_dataset)
        shuffle = bs < len(self.tokenized_dataset)
        return torch_data.DataLoader(self.tokenized_dataset, batch_size=bs, shuffle=shuffle, pin_memory=self.pin_memory)

    def val_dataloader(self) -> torch_data.DataLoader:
        bs = self.batch_size if self.batch_size > 0 else len(self.tokenized_dataset)
        return torch_data.DataLoader(self.tokenized_dataset, batch_size=bs, shuffle=False, pin_memory=self.pin_memory)

    def test_dataloader(self) -> torch_data.DataLoader:
        bs = self.batch_size if self.batch_size > 0 else len(self.tokenized_dataset)
        return torch_data.DataLoader(self.tokenized_dataset, batch_size=bs, shuffle=False, pin_memory=self.pin_memory)
