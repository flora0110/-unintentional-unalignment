# goodreads_experiments/data/goodreads_dpo_datamodule.py
from typing import Dict
import datasets
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from common.data.modules import DataModule

class GoodreadsSingleOutputTokenDataModule(DataModule):
    """
    Dataset item (after tokenize):
      - prompt_input_ids: [B, Lp]
      - prompt_attention_mask: [B, Lp]
      - chosen_ids: [B, Lc]
      - rejected_ids: [B, Lr]
      - ref_logp_chosen: [B]     # sum over chosen response tokens
      - ref_logp_rejected: [B]   # sum over rejected response tokens
    """
    def __init__(self, path: str, model: nn.Module, tokenizer,
                 num_train_samples: int = -1, batch_size: int = -1,
                 pin_memory: bool = False, device=torch.device("cpu"),
                 load_dataset_to_device=None, random_seed: int = -1):
        self.path = path
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.num_train_samples = num_train_samples
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.device = device
        self.load_dataset_to_device = load_dataset_to_device
        self.random_seed = random_seed

    def setup(self):
        ds = datasets.load_dataset("json", data_files=self.path, split="train")

        # sub-sample if needed
        if self.random_seed and self.random_seed > 0:
            g = torch.Generator().manual_seed(self.random_seed)
            perm = torch.randperm(len(ds), generator=g)
        else:
            perm = torch.randperm(len(ds))
        if self.num_train_samples and 0 < self.num_train_samples < len(ds):
            ds = ds.select(perm[: self.num_train_samples])

        # tokenize
        ds = ds.map(self._tokenize_example, batched=True, remove_columns=ds.column_names)
        ds.set_format(type="torch")

        # move model to device & eval
        self.model.to(self.device).eval()

        # add frozen reference log-probs (computed once at setup)
        ds = ds.map(self._add_reference_log_probs, batched=True, batch_size=self.batch_size)
        self.tokenized_dataset = ds

    def _tokenize_example(self, batch: Dict) -> Dict:
        # We assume 'prompt', 'chosen', 'rejected' are strings.
        prompt_enc = self.tokenizer(batch["prompt"], add_special_tokens=True, padding=True, return_tensors="pt")
        chosen_enc = self.tokenizer(batch["chosen"], add_special_tokens=False, padding=False, return_tensors="pt")
        rejected_enc = self.tokenizer(batch["rejected"], add_special_tokens=False, padding=False, return_tensors="pt")

        return {
            "prompt_input_ids": prompt_enc["input_ids"],
            "prompt_attention_mask": prompt_enc["attention_mask"],
            "chosen_ids": chosen_enc["input_ids"],
            "rejected_ids": rejected_enc["input_ids"],
        }

    @torch.no_grad()
    def _sequence_logprob(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                          resp_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(resp_ids | prompt) for a causal LM:
          - Build full = [prompt, resp]
          - logits_for_next = logits[:, :-1, :]
          - labels = full[:, 1:]
          - Only sum labels within response span.
        Shapes:
          input_ids: [B, Lp], attention_mask: [B, Lp], resp_ids: [B, Lr]
        Return:
          logp_sum: [B]
        """
        B = input_ids.size(0)
        full = torch.cat([input_ids, resp_ids], dim=1)           # [B, Lp+Lr]
        attn = torch.cat([attention_mask,
                          torch.ones((B, resp_ids.size(1)), dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)

        outputs = self.model(input_ids=full.to(self.device), attention_mask=attn.to(self.device))
        logits = outputs.logits  # [B, Lp+Lr, V]

        # shift for next-token prediction
        logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)      # [B, Lp+Lr-1, V]
        labels = full[:, 1:]                                     # [B, Lp+Lr-1]

        # response starts at index Lp in "full"; in "labels"/"logprobs" coordinates, response tokens map to positions [Lp-1 ... Lp+Lr-2]
        Lp = input_ids.size(1)
        start = Lp - 1
        end = start + resp_ids.size(1)                           # exclusive

        resp_token_logprobs = logprobs[torch.arange(B).unsqueeze(1),
                                       torch.arange(start, end).unsqueeze(0).to(logprobs.device),
                                       resp_ids.to(self.device)]
        # sum over response tokens
        return resp_token_logprobs.sum(dim=1)

    @torch.no_grad()
    def _add_reference_log_probs(self, batch: Dict) -> Dict:
        # put tensors on device
        prompt_ids = batch["prompt_input_ids"].to(self.device)
        prompt_mask = batch["prompt_attention_mask"].to(self.device)
        chosen_ids = batch["chosen_ids"].to(self.device)
        rejected_ids = batch["rejected_ids"].to(self.device)

        ref_logp_chosen = self._sequence_logprob(prompt_ids, prompt_mask, chosen_ids)
        ref_logp_rejected = self._sequence_logprob(prompt_ids, prompt_mask, rejected_ids)

        batch["ref_logp_chosen"] = ref_logp_chosen
        batch["ref_logp_rejected"] = ref_logp_rejected
        return batch

    def train_dataloader(self):
        bs = self.batch_size if self.batch_size > 0 else len(self.tokenized_dataset)
        shuffle = bs < len(self.tokenized_dataset)
        return torch.utils.data.DataLoader(self.tokenized_dataset, batch_size=bs, shuffle=shuffle, pin_memory=self.pin_memory)

    def val_dataloader(self):
        bs = self.batch_size if self.batch_size > 0 else len(self.tokenized_dataset)
        return torch.utils.data.DataLoader(self.tokenized_dataset, batch_size=bs, shuffle=False, pin_memory=self.pin_memory)

    def test_dataloader(self):
        bs = self.batch_size if self.batch_size > 0 else len(self.tokenized_dataset)
        return torch.utils.data.DataLoader(self.tokenized_dataset, batch_size=bs, shuffle=False, pin_memory=self.pin_memory)
