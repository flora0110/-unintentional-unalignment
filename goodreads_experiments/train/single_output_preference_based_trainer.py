from typing import List

import torch
import torch.nn.functional as F

from common.evaluation.evaluators.evaluator import VoidEvaluator
from common.train.trainer import Trainer


class SingleOutputPreferenceBasedTrainer(Trainer):
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

    @torch.no_grad()
    def _seq_logprob(self, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor, resp_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(resp | prompt) under *current* policy (used only for logging if needed).
        Not used for reference; reference is passed from batch.
        """
        B = prompt_ids.size(0)
        device = prompt_ids.device
        full = torch.cat([prompt_ids, resp_ids], dim=1)                  # [B, Lp+Lr]
        attn = torch.cat([prompt_mask, torch.ones((B, resp_ids.size(1)), dtype=prompt_mask.dtype, device=device)], dim=1)

        outs = self.model(input_ids=full, attention_mask=attn)
        logits = outs.logits                                             # [B, L, V]
        logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)              # predict next token
        labels = full[:, 1:]                                             # gold next

        Lp = prompt_ids.size(1)
        start = Lp - 1
        end = start + resp_ids.size(1)
        tok_lp = logprobs[torch.arange(B, device=device).unsqueeze(1),
                          torch.arange(start, end, device=device).unsqueeze(0),
                          resp_ids]
        return tok_lp.sum(dim=1)

    def _seq_logprob_policy(self, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor, resp_ids: torch.Tensor) -> torch.Tensor:
        """Same as _seq_logprob but kept separate for clarity; this is the policy term used in the loss."""
        B = prompt_ids.size(0)
        device = prompt_ids.device
        full = torch.cat([prompt_ids, resp_ids], dim=1)
        attn = torch.cat([prompt_mask, torch.ones((B, resp_ids.size(1)), dtype=prompt_mask.dtype, device=device)], dim=1)

        outs = self.model(input_ids=full, attention_mask=attn)
        logits = outs.logits
        logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)
        Lp = prompt_ids.size(1)
        start = Lp - 1
        end = start + resp_ids.size(1)
        tok_lp = logprobs[torch.arange(B, device=device).unsqueeze(1),
                          torch.arange(start, end, device=device).unsqueeze(0),
                          resp_ids]
        return tok_lp.sum(dim=1)

    def batch_update(self, batch_num, batch, total_num_batches):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        preferred_output_ids = batch["preferred_output_ids"].to(self.device)
        dispreferred_output_ids = batch["dispreferred_output_ids"].to(self.device)
        ref_preferred_logprobs = batch["ref_preferred_logprobs"].to(self.device)
        ref_dispreferred_logprobs = batch["ref_dispreferred_logprobs"].to(self.device)
        unembedding_weights_pre_update = torch.clone(self.model.get_output_embeddings().weight.data.detach())

        # policy logp on sequences
        pol_logp_chosen = self._seq_logprob_policy(prompt_ids, prompt_mask, chosen_ids)
        pol_logp_rejected = self._seq_logprob_policy(prompt_ids, prompt_mask, rejected_ids)
        # 

        # outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # all_logits = outputs.logits
        # output_logits = all_logits[:, -1, :]
        # output_logprobs = F.log_softmax(output_logits, dim=1)

        # preferred_logprobs = output_logprobs[torch.arange(output_logprobs.size(0)), preferred_output_ids]
        # dispreferred_logprobs = output_logprobs[torch.arange(output_logprobs.size(0)), dispreferred_output_ids]

        # if self.objective == "dpo":
        #     loss = self.__compute_dpo_loss(preferred_logprobs, dispreferred_logprobs, ref_preferred_logprobs, ref_dispreferred_logprobs)
        # elif self.objective == "ipo":
        #     loss = self.__compute_ipo_loss(preferred_logprobs, dispreferred_logprobs, ref_preferred_logprobs, ref_dispreferred_logprobs)
        # elif self.objective == "cross_entropy":
        #     loss = - preferred_logprobs.mean()

        if self.objective == "dpo":
            loss = self.__compute_dpo_loss(pol_logp_chosen, pol_logp_rejected, ref_logp_chosen, ref_logp_rejected)
        elif self.objective == "ipo":
            loss = self.__compute_ipo_loss(pol_logp_chosen, pol_logp_rejected, ref_logp_chosen, ref_logp_rejected)
        elif self.objective == "cross_entropy":
            # CE on chosen sequence average per token (optional); here we keep simple sequence sum mean over batch
            loss = (- pol_logp_chosen).mean()

        if self.gradient_accumulation > 0:
            loss = loss / self.gradient_accumulation
        loss.backward()

        do_accumulated_grad_update = (batch_num + 1) % self.gradient_accumulation == 0 or batch_num == total_num_batches - 1
        if self.gradient_accumulation <= 0 or do_accumulated_grad_update:
            self.optimizer.step()
            self.optimizer.zero_grad()

        output_dict = {
            # "train loss": loss.item(),
            # "output logits": output_logits.detach(),
            # "output logprobs": output_logprobs.detach(),
            # "input ids": input_ids,
            # "preferred output ids": preferred_output_ids,
            # "dispreferred output ids": dispreferred_output_ids,
            # "preferred logit": output_logits[torch.arange(output_logprobs.size(0)), preferred_output_ids].detach().mean().item(),
            # "dispreferred logit": output_logits[torch.arange(output_logprobs.size(0)), dispreferred_output_ids].detach().mean().item(),
            # "preferred prob": torch.exp(preferred_logprobs).detach().mean().item(),
            # "dispreferred prob": torch.exp(dispreferred_logprobs).detach().mean().item(),
            # "preferred logprob change": (preferred_logprobs - ref_preferred_logprobs).detach().mean().item(),
            # "dispreferred logprob change": (dispreferred_logprobs - ref_dispreferred_logprobs).detach().mean().item(),
            # "unembedding weights": unembedding_weights_pre_update,
            # "hidden representations": outputs.hidden_states[-1][:, -1, :].detach()
            "train loss": loss.item(),
            # 序列級
            "seq/pol_logp_chosen": pol_seq_logp_chosen.detach(),
            "seq/pol_logp_rejected": pol_seq_logp_rejected.detach(),
            "seq/ref_logp_chosen": ref_seq_logp_chosen.detach(),
            "seq/ref_logp_rejected": ref_seq_logp_rejected.detach(),
            "seq/pol_logp_chosen_change":    (pol_seq_logp_chosen   - ref_seq_logp_chosen  ).detach().mean().item(),
            "seq/pol_logp_rejected_change": (pol_seq_logp_rejected - ref_seq_logp_rejected).detach().mean().item(),
            "chosen_item_ids":   batch["chosen_item_ids"].to(self.device),
            "rejected_item_ids": batch["rejected_item_ids"].to(self.device),

            "seq/logp_margin": (pol_seq_logp_chosen - pol_seq_logp_rejected).detach(),
            "seq/ref_logp_margin": (ref_seq_logp_chosen - ref_seq_logp_rejected).detach(),
            "seq/margin_delta": ((pol_seq_logp_chosen - pol_seq_logp_rejected) - (ref_seq_logp_chosen - ref_seq_logp_rejected)).detach(),
            

            # resp0 token（給 callback 做 token top-k）
            "resp0_logprobs": resp0_logprobs.detach(),    # [B, V]
            "prompt_input_ids": prompt_ids,               # [B, Lp]
            "resp0_pref_ids": resp0_pref_ids,             # [B]
            "resp0_disp_ids": resp0_disp_ids,             # [B]
            "unembedding weights": unembedding_weights_pre_update,
            "resp0_hidden": resp0_hidden.detach(),

            # 也可回傳 chosen/rejected 的 id，callback 若要其它分析用得到
            "chosen_ids": chosen_ids,
            "rejected_ids": rejected_ids,

            "prompt_input_ids": prompt_ids,
        }
        self.__add_other_tokens_logit_prob_info(output_dict, output_logits, output_logprobs)
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
