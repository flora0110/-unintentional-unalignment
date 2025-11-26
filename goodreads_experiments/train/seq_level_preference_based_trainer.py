from typing import List

import torch
import torch.nn.functional as F

from common.evaluation.evaluators.evaluator import VoidEvaluator
from common.train.trainer import Trainer


class SeqLevelPreferenceBasedTrainer(Trainer):
    """
    Trainer for preference-based objective. Currently supports DPO, IPO, and cross entropy using the preferred outputs.
    """

    def __init__(self, model, tokenizer, optimizer, kl_coeff: float = 0.1, objective: str = "dpo", train_evaluator=VoidEvaluator(),
                 val_evaluator=VoidEvaluator(), callback=None, device=torch.device("cpu"), track_logits_for_tokens: List[str] = None,
                 gradient_accumulation: int = -1):
        super().__init__(model, optimizer, train_evaluator, val_evaluator, callback, device)
        self.tokenizer = tokenizer
        self.kl_coeff = kl_coeff
        self.objective = objective
        if self.objective not in ["dpo", "ipo", "cross_entropy"]:
            raise ValueError(f"Objective {self.objective} is not supported. Must be one of ['dpo', 'ipo', 'cross_entropy']")

        self.track_logits_for_tokens = track_logits_for_tokens
        self.gradient_accumulation = gradient_accumulation

    # ---- helper: 對一個 (prompt + answer) 序列批次算 seq-level logprob + hidden 統計 ----
    def _forward_and_seq_logprob(self, input_ids, attention_mask, answer_mask):
        """
        計算：
          - seq_logprobs[b] = sum_{t in answer tokens} log p(y_t | prefix)
          - mean_logprobs[b] = seq_logprobs[b] / #answer_tokens[b]
        並且回傳：
          - last_logits:   每個樣本「最後一個 answer token」位置的 logits
          - last_logprobs: 上述 logits 經 softmax 後的 log 機率
          - last_hidden:   每個樣本「最後一個 answer token」位置的 hidden state
          - sum_hidden:    每個樣本在 answer 區段上 hidden 的總和（之後算 CHES 用）
        其中 answer_mask 是 [B, T]，在「answer token 位置」為 1，其餘為 0。
        """
        # input_ids, attention_mask, answer_mask: [B, T]
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits = outputs.logits                 # [B, T, V]
        hidden = outputs.hidden_states[-1]      # [B, T, H]

        B, T, V = logits.shape
        _, _, H = hidden.shape
        device = logits.device

        # ----------------------------------------
        # 1) seq-level logprob：對 answer 區段的 token 做 sum / mean
        # ----------------------------------------
        # logits[t] 預測的是 input_ids[t+1]，所以要 shift 一格
        logits_shifted = logits[:, :-1, :]      # [B, T-1, V]
        labels = input_ids[:, 1:]               # [B, T-1]
        logprobs_shifted = F.log_softmax(logits_shifted, dim=-1)

        # token_logprobs[b, t] = log p(input_ids[b, t+1] | prefix up to t)
        token_logprobs = logprobs_shifted.gather(
            dim=-1,
            index=labels.unsqueeze(-1),
        ).squeeze(-1)                           # [B, T-1]

        # 對「答案部分」且非 pad 的位置做 sum
        # answer_mask / attention_mask 是 [B, T]，也要對齊 shift 後的長度
        mask_shifted = answer_mask[:, 1:] * attention_mask[:, 1:]    # [B, T-1]
        seq_logprobs = (token_logprobs * mask_shifted).sum(dim=-1)   # [B]

        # 每個 sample 的 answer token 數，避免除以 0
        num_answer_tokens = mask_shifted.sum(dim=-1).clamp(min=1)    # [B]
        mean_logprobs = seq_logprobs / num_answer_tokens             # [B]

        # ----------------------------------------
        # 2) answer 區段 hidden 統計（CHES / last hidden）
        # ----------------------------------------
        # 使用未 shift 的 mask：哪些位置是 answer token（且非 pad）
        ans_mask_full = (answer_mask * attention_mask).to(hidden.dtype)   # [B, T]

        # sum_hidden[b] = Σ_{t in answer} h[b, t]
        sum_hidden = (hidden * ans_mask_full.unsqueeze(-1)).sum(dim=1)    # [B, H]

        # 找出每個 sample「最後一個 answer token」的位置 index
        # positions = 0,1,...,T-1
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # [B, T]
        ans_bool = ans_mask_full > 0                                         # [B, T]

        # 理論上 datamodule 已確保至少一個 answer token，但這裡做個 fallback：
        # 若某個樣本沒有 answer token，就用最後一個非 pad 位置
        has_answer = ans_bool.any(dim=1)                                     # [B]
        last_ans_idx = (positions * ans_bool).max(dim=1).values             # [B]
        last_nonpad_idx = attention_mask.sum(dim=1) - 1                      # [B]

        last_idx = torch.where(has_answer, last_ans_idx, last_nonpad_idx)   # [B]
        last_idx = last_idx.long()

        # 取出最後 answer 位置的 hidden / logits / logprobs
        batch_indices = torch.arange(B, device=device)

        last_hidden = hidden[batch_indices, last_idx]              # [B, H]
        last_logits = logits[batch_indices, last_idx]              # [B, V]
        last_logprobs = F.log_softmax(last_logits, dim=-1)         # [B, V]

        return (
            seq_logprobs,      # [B]
            mean_logprobs,     # [B]
            last_logits,       # [B, V]
            last_logprobs,     # [B, V]
            last_hidden,       # [B, H]
            sum_hidden,        # [B, H]
        )

    def batch_update(self, batch_num, batch, total_num_batches):
        # ---- 1. 取出 batch 欄位 ----
        pref_input_ids = batch["pref_input_ids"].to(self.device)        # [B, T]
        pref_att_mask  = batch["pref_attention_mask"].to(self.device)   # [B, T]
        pref_ans_mask  = batch["pref_answer_mask"].to(self.device)      # [B, T]

        rej_input_ids = batch["rej_input_ids"].to(self.device)          # [B, T]
        rej_att_mask  = batch["rej_attention_mask"].to(self.device)     # [B, T]
        rej_ans_mask  = batch["rej_answer_mask"].to(self.device)        # [B, T]

        # 參考 policy 的 seq-level log p(y|x)（在 datamodule 先算好的）
        ref_pref_logprobs = batch["ref_pref_logprobs"].to(self.device)  # [B]
        ref_rej_logprobs  = batch["ref_rej_logprobs"].to(self.device)   # [B]

        # unembedding 在 update 前先存一份（用來看 token logit/embedding drift）
        unembedding_weights_pre_update = torch.clone(
            self.model.get_output_embeddings().weight.data.detach()
        )

        # ---- 2. 前向：pref / rej 各跑一次，算 seq-level logprob + hidden 統計 ----
        (
            pref_seq_logprobs,     # [B]  sum over answer tokens
            pref_mean_logprobs,    # [B]  per-answer-token avg
            pref_last_logits,      # [B, V] 最後一個 answer token 的 logits
            pref_last_logprobs,    # [B, V]
            pref_last_hidden,      # [B, H]
            pref_sum_hidden,       # [B, H] answer 區段 hidden sum
        ) = self._forward_and_seq_logprob(
            pref_input_ids, pref_att_mask, pref_ans_mask
        )

        (
            rej_seq_logprobs,
            rej_mean_logprobs,
            rej_last_logits,
            rej_last_logprobs,
            rej_last_hidden,
            rej_sum_hidden,
        ) = self._forward_and_seq_logprob(
            rej_input_ids, rej_att_mask, rej_ans_mask
        )

        # DPO/IPO 的「logprobs」就是整段 answer 的 seq_logprobs
        preferred_logprobs    = pref_seq_logprobs  # [B]
        dispreferred_logprobs = rej_seq_logprobs   # [B]

        # ---- 3. 算 loss（seq-level 版）----
        if self.objective == "dpo":
            loss = self.__compute_dpo_loss(
                preferred_logprobs,
                dispreferred_logprobs,
                ref_pref_logprobs,
                ref_rej_logprobs,
            )
        elif self.objective == "ipo":
            loss = self.__compute_ipo_loss(
                preferred_logprobs,
                dispreferred_logprobs,
                ref_pref_logprobs,
                ref_rej_logprobs,
            )
        elif self.objective == "cross_entropy":
            # seq-level CE: maximize log p(y_pref | x)
            loss = -preferred_logprobs.mean()
        else:
            raise ValueError(f"Unsupported objective: {self.objective}")

        # ---- 4. 反向傳播 + gradient accumulation ----
        if self.gradient_accumulation > 0:
            loss_to_backprop = loss / self.gradient_accumulation
        else:
            loss_to_backprop = loss

        loss_to_backprop.backward()

        if self.gradient_accumulation > 0:
            do_accumulated_grad_update = (
                ((batch_num + 1) % self.gradient_accumulation == 0)
                or (batch_num == total_num_batches - 1)
            )
        else:
            do_accumulated_grad_update = True

        if do_accumulated_grad_update:
            self.optimizer.step()
            self.optimizer.zero_grad()

        # ---- 5. 組 output_dict，給 evaluator / callbacks 用 ----

        # per-token 平均機率（對整個 batch 再取 mean）
        pref_prob_batch = torch.exp(pref_mean_logprobs).detach()  # [B]
        rej_prob_batch  = torch.exp(rej_mean_logprobs).detach()   # [B]

        output_dict = {
            "train loss": loss.item(),

            # 之後 callback 想要還原文字 / 算 edit distance 用
            "preferred output ids": pref_input_ids.detach().cpu(),  # [B, T]
            "dispreferred output ids": rej_input_ids.detach().cpu(),# [B, T]
            "pref_ans_mask": pref_ans_mask.detach().cpu(),          # [B, T]
            "rej_ans_mask":  rej_ans_mask.detach().cpu(),           # [B, T]

            # batch 平均的指標
            "preferred prob": pref_prob_batch.mean().item(),
            "dispreferred prob": rej_prob_batch.mean().item(),
            "preferred logprob change": (preferred_logprobs - ref_pref_logprobs).detach().mean().item(),
            "dispreferred logprob change": (dispreferred_logprobs - ref_rej_logprobs).detach().mean().item(),

            # 每個 sample 的 seq-level logprob，用來畫分佈 / per-sample 分析
            "preferred_logprobs_list": preferred_logprobs.detach().cpu(),   # [B]
            "dispreferred_logprobs_list": dispreferred_logprobs.detach().cpu(), # [B]
            "ref_pref_logprobs_list": ref_pref_logprobs.detach().cpu(),     # [B]
            "ref_rej_logprobs_list":  ref_rej_logprobs.detach().cpu(),      # [B]

            # CHES / hidden drift 用
            "pref_sum_hidden": pref_sum_hidden.detach().cpu(),   # [B, H]
            "rej_sum_hidden":  rej_sum_hidden.detach().cpu(),    # [B, H]
            "h_pref_last":     pref_last_hidden.detach().cpu(),  # [B, H]
            "h_rej_last":      rej_last_hidden.detach().cpu(),   # [B, H]

            # 和原版對齊：用 preferred 的最後 answer token 當「代表性」logits/hidden
            "unembedding weights": unembedding_weights_pre_update,       # [V, H]（或依模型而定）
            "hidden representations": pref_last_hidden.detach().cpu(),   # [B, H]
        }

        # # 如果你還是想追蹤某些 token 的 logit/prob，就用 preferred 分支的最後一個 answer token 的分佈
        # self.__add_other_tokens_logit_prob_info(
        #     output_dict,
        #     pref_last_logits.detach().cpu(),      # [B, V]
        #     pref_last_logprobs.detach().cpu(),    # [B, V]
        # )

        return output_dict

    def __compute_dpo_loss(self, preferred_logprobs, dispreferred_logprobs, ref_preferred_logprobs, ref_dispreferred_logprobs):
        log_prob_ratio = preferred_logprobs - dispreferred_logprobs
        ref_log_prob_ratio = ref_preferred_logprobs - ref_dispreferred_logprobs
        return - F.logsigmoid(self.kl_coeff * (log_prob_ratio - ref_log_prob_ratio)).mean()

    def __compute_ipo_loss(self, preferred_logprobs, dispreferred_logprobs, ref_preferred_logprobs, ref_dispreferred_logprobs):
        log_prob_ratio = preferred_logprobs - dispreferred_logprobs
        ref_log_prob_ratio = ref_preferred_logprobs - ref_dispreferred_logprobs
        return ((log_prob_ratio - ref_log_prob_ratio - 1 / (2 * self.kl_coeff)) ** 2).mean()

    def __add_other_tokens_logit_prob_info(self, output_dict: dict, output_logits: torch.Tensor, output_logprobs: torch.Tensor):
        if self.track_logits_for_tokens is None:
            return

        for token in self.track_logits_for_tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            output_dict[f"{token} logit"] = output_logits[torch.arange(output_logprobs.size(0)), token_id].detach().mean().item()
            output_dict[f"{token} prob"] = torch.exp(output_logprobs[torch.arange(output_logprobs.size(0)), token_id]).detach().mean().item()
