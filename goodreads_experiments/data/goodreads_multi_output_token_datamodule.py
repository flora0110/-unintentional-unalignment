import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os
from common.data.modules import DataModule


class GoodreadsMultiOutputTokenDataModule(DataModule):
    """
    Seq-level DPO / SFT 用的 Goodreads datamodule。

    假設原始資料格式：
    {
        "prompt": str,   # 已含 Instruction / Input / "### Response:"
        "chosen": str,   # 比如 "\"The Undoing (Call of Crows, #2)\""
        "rejected": str,
    }
    """

    def __init__(
        self,
        path: str,
        model: nn.Module,
        tokenizer,
        batch_size: int,
        max_length: int = 512,
        pin_memory: bool = False,
        device: torch.device = torch.device("cpu"),
        load_dataset_to_device=None,
        ref_cache_dir: str = None,
        use_ref_cache: bool = True,
    ):
        self.path = path
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.pin_memory = pin_memory
        self.device = device
        self.load_dataset_to_device = load_dataset_to_device

        # ★ 新增：ref cache 設定
        self.use_ref_cache = use_ref_cache
        if ref_cache_dir is None:
            # 預設：在 dataset 檔名旁邊開一個資料夾
            # 例如 data_files/goodreads/train.json → data_files/goodreads/train.json_ref_cache
            self.ref_cache_dir = self.path + "_ref_cache"
        else:
            self.ref_cache_dir = ref_cache_dir
        # 保險：確保有 pad_token，並設成左 pad（跟你 persona 版一致）
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ---------- high-level ----------

    # def setup(self):
    #     # 1) 讀 json list
    #     self.dataset = datasets.load_dataset("json", data_files=self.path, split="train")

    #     # 2) 這裡其實不太需要額外 prepare，prompt / chosen / rejected 已經在正確欄位
    #     #    如果你之後想改 format，可以在這裡 map 一層。

    #     # 3) tokenize 成 pref/rej 兩個完整序列 + answer mask
    #     self.tokenized_dataset = self.__tokenize_dataset(self.dataset)

    #     # 4) 用 reference model 算 seq-level logprob
    #     self.model.to(self.device)
    #     self.model.eval()
    #     self.tokenized_dataset = self.tokenized_dataset.map(
    #         self.__add_reference_log_probs,
    #         batched=True,
    #         batch_size=self.batch_size,
    #     )

    def setup(self):
        # 1) 原始 json dataset
        self.dataset = datasets.load_dataset("json", data_files=self.path, split="train")

        # 2) 先做 tokenize（不含 ref）
        tokenized = self.__tokenize_dataset(self.dataset)

        # 3) 如果開啟 cache，且 cache 資料夾存在 → 直接 load
        if self.use_ref_cache and os.path.exists(self.ref_cache_dir):
            print(f"[GoodreadsMultiOutputTokenDataModule] Loading cached tokenized dataset with ref logprobs from {self.ref_cache_dir}")
            cached = datasets.load_from_disk(self.ref_cache_dir)

            # 重新設 format（load_from_disk 不會記得 set_format）
            format_kwargs = {}
            if self.load_dataset_to_device is not None:
                format_kwargs["device"] = self.load_dataset_to_device

            cached.set_format(
                type="torch",
                columns=[
                    "pref_input_ids", "pref_attention_mask", "pref_answer_mask",
                    "rej_input_ids",  "rej_attention_mask",  "rej_answer_mask",
                    # ref_pref_logprobs / ref_rej_logprobs 可以是 float list，讓 DataLoader 自動轉 tensor
                ],
                **format_kwargs,
            )
            self.tokenized_dataset = cached
            return

        # 4) 沒有 cache → 用 reference model 算 ref logprobs，然後存起來
        self.model.to(self.device)
        self.model.eval()

        tokenized_with_ref = tokenized.map(
            self.__add_reference_log_probs,
            batched=True,
            batch_size=self.batch_size,
        )

        # 存成「含 ref」的完整 tokenized dataset
        if self.use_ref_cache:
            os.makedirs(self.ref_cache_dir, exist_ok=True)
            print(f"[GoodreadsMultiOutputTokenDataModule] Saving tokenized dataset with ref logprobs to {self.ref_cache_dir}")
            tokenized_with_ref.save_to_disk(self.ref_cache_dir)

        self.tokenized_dataset = tokenized_with_ref


    # ---------- tokenization ----------

    def __tokenize_example(self, example: dict) -> dict:
        """
        example keys: "prompt", "chosen", "rejected"
        這裡會產生：
          pref_input_ids, pref_attention_mask, pref_answer_mask
          rej_input_ids,  rej_attention_mask,  rej_answer_mask
        """
        prompts = example["prompt"]
        chosens = example["chosen"]
        rejecteds = example["rejected"]

        # 1) 先 tokenize prompt 本身，取得 prompt 長度（之後做 answer_mask 用）
        tok_prompt = self.tokenizer(
            prompts,
            add_special_tokens=False,   # prompt 本身已經包含你想要的格式
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        prompt_lengths = [len(ids) for ids in tok_prompt["input_ids"]]

        # 2) tokenize prompt + chosen / rejected
        pref_texts = [p + c for p, c in zip(prompts, chosens)]
        rej_texts  = [p + r for p, r in zip(prompts, rejecteds)]

        pref_tok = self.tokenizer(
            pref_texts,
            add_special_tokens=False,
            padding="max_length",   # ★ 這裡改掉
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        rej_tok = self.tokenizer(
            rej_texts,
            add_special_tokens=False,
            padding="max_length",   # ★ 這裡也改掉
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )


        B, T = pref_tok["input_ids"].shape
        pref_answer_mask = torch.zeros((B, T), dtype=torch.long)
        rej_answer_mask  = torch.zeros((B, T), dtype=torch.long)

       # 3) 對每個樣本：prompt 之後、且非 pad 的 token 視為 answer
        for i, plen in enumerate(prompt_lengths):
            plen = min(plen, T)

            
           

            # preferred
            valid_pref = pref_tok["attention_mask"][i] == 1   # [T] boolean
            seq_len_pref = valid_pref.shape[0]
            # 左右兩邊都是從 plen: 到結尾，長度一致
            pref_answer_mask[i, plen:seq_len_pref][valid_pref[plen:]] = 1

            # rejected
            valid_rej = rej_tok["attention_mask"][i] == 1     # [T] boolean
            seq_len_rej = valid_rej.shape[0]
            rej_answer_mask[i, plen:seq_len_rej][valid_rej[plen:]] = 1
        return {
            "pref_input_ids":      pref_tok["input_ids"],
            "pref_attention_mask": pref_tok["attention_mask"],
            "pref_answer_mask":    pref_answer_mask,
            "rej_input_ids":       rej_tok["input_ids"],
            "rej_attention_mask":  rej_tok["attention_mask"],
            "rej_answer_mask":     rej_answer_mask,
        }

    # def __tokenize_dataset(self, dataset: datasets.Dataset) -> datasets.Dataset:
    #     dataset = dataset.map(self.__tokenize_example, batched=True)

    #     # 讓 DataLoader 直接回傳 torch.Tensor
    #     dataset.set_format(
    #         type="torch",
    #         columns=[
    #             "pref_input_ids", "pref_attention_mask", "pref_answer_mask",
    #             "rej_input_ids",  "rej_attention_mask",  "rej_answer_mask",
    #         ],
    #     )

    #     if self.load_dataset_to_device is not None:
    #         dataset = dataset.map(
    #             lambda batch: {k: v.to(self.load_dataset_to_device) for k, v in batch.items()},
    #             batched=True,
    #         )
    #     return dataset

    def __tokenize_dataset(self, dataset: datasets.Dataset) -> datasets.Dataset:
        dataset = dataset.map(self.__tokenize_example, batched=True)

        # # 讓 DataLoader 直接回傳 torch.Tensor；如果指定了 device，就一起設進去
        # format_kwargs = {}
        # if self.load_dataset_to_device is not None:
        #     format_kwargs["device"] = self.load_dataset_to_device

        dataset.set_format(
            type="torch",
            columns=[
                "pref_input_ids", "pref_attention_mask", "pref_answer_mask",
                "rej_input_ids",  "rej_attention_mask",  "rej_answer_mask",
            ],
            # **format_kwargs,
        )

        return dataset


    # ---------- reference logprobs ----------

    def __seq_logprob(self, input_ids, attention_mask, answer_mask):
        """
        計算 seq-level log π_ref(y | x) = sum over answer tokens log p(token_t | context)。
        """
        with torch.no_grad():
            # 把 model 傳進 DataModule，讓 DataModule 用這個 model 預先算 reference logprob
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]               # [B, T-1, V]
            labels = input_ids[:, 1:]                        # [B, T-1]

            logprobs = F.log_softmax(logits, dim=-1)
            token_logprobs = logprobs.gather(
                dim=-1, index=labels.unsqueeze(-1)
            ).squeeze(-1)                                    # [B, T-1]

            # 只對「answer」且非 pad 的位置做 sum
            mask = answer_mask[:, 1:] * attention_mask[:, 1:]
            seq_logprobs = (token_logprobs * mask).sum(dim=-1)  # [B]
        return seq_logprobs

    # def __add_reference_log_probs(self, batch: dict) -> dict:
    #     pref_ids = batch["pref_input_ids"].to(self.device)
    #     pref_att = batch["pref_attention_mask"].to(self.device)
    #     pref_ans = batch["pref_answer_mask"].to(self.device)

    #     rej_ids  = batch["rej_input_ids"].to(self.device)
    #     rej_att  = batch["rej_attention_mask"].to(self.device)
    #     rej_ans  = batch["rej_answer_mask"].to(self.device)

    #     ref_pref = self.__seq_logprob(pref_ids, pref_att, pref_ans)
    #     ref_rej  = self.__seq_logprob(rej_ids,  rej_att,  rej_ans)

    #     batch["ref_pref_logprobs"] = ref_pref
    #     batch["ref_rej_logprobs"]  = ref_rej
    #     return batch

    def __add_reference_log_probs(self, batch: dict) -> dict:
        # 將 HF map 給的 batch 欄位安全轉成 tensor 並搬到 self.device
        def to_tensor(x, dtype=torch.long):
            if isinstance(x, torch.Tensor):
                return x.to(self.device)

            # x 是 list，且元素是 tensor → 用 stack
            if isinstance(x, list) and len(x) > 0 and isinstance(x[0], torch.Tensor):
                return torch.stack(x, dim=0).to(self.device).to(dtype)

            # 其他情況（list of list / numpy / scalar）直接丟給 torch.tensor
            return torch.tensor(x, dtype=dtype, device=self.device)

        # 這裡 batch[...] 可能是 list-of-tensor 或 list-of-list，都交給 to_tensor 處理
        pref_ids = to_tensor(batch["pref_input_ids"], dtype=torch.long)
        pref_att = to_tensor(batch["pref_attention_mask"], dtype=torch.long)
        pref_ans = to_tensor(batch["pref_answer_mask"], dtype=torch.long)

        rej_ids  = to_tensor(batch["rej_input_ids"], dtype=torch.long)
        rej_att  = to_tensor(batch["rej_attention_mask"], dtype=torch.long)
        rej_ans  = to_tensor(batch["rej_answer_mask"], dtype=torch.long)

        # 用 reference model 算 seq-level log π_ref
        ref_pref = self.__seq_logprob(pref_ids, pref_att, pref_ans)   # [B]
        ref_rej  = self.__seq_logprob(rej_ids,  rej_att,  rej_ans)   # [B]

        # 回傳時轉回 CPU list，方便 datasets 存到 Arrow
        batch["ref_pref_logprobs"] = ref_pref.detach().cpu().tolist()
        batch["ref_rej_logprobs"]  = ref_rej.detach().cpu().tolist()
        return batch


    # ---------- dataloaders ----------

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        batch_size = self.batch_size if self.batch_size > 0 else len(self.tokenized_dataset)
        shuffle = batch_size < len(self.tokenized_dataset)
        return torch.utils.data.DataLoader(
            self.tokenized_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        batch_size = self.batch_size if self.batch_size > 0 else len(self.tokenized_dataset)
        return torch.utils.data.DataLoader(
            self.tokenized_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        batch_size = self.batch_size if self.batch_size > 0 else len(self.tokenized_dataset)
        return torch.utils.data.DataLoader(
            self.tokenized_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
