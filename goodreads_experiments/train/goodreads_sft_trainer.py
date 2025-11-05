# unintentional-unalignment/goodreads_experiments/train/goodreads_sft_trainer.py
from __future__ import annotations
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from common.train.trainer import Trainer
from common.evaluation.evaluators import TrainEvaluator, Evaluator
from common.train.fit_output import FitOutput

class GoodreadsSFTTrainer(Trainer):
    """
    Minimal SFT trainer: standard causal-lm CE loss with label masking.
    Expects batches with {input_ids, attention_mask, labels}.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        optimizer: torch.optim.Optimizer,
        train_evaluator: TrainEvaluator,
        val_evaluator: Evaluator,
        callback,
        device,
        grad_accum_steps: int = -1,
    ):
        super().__init__(model=model, optimizer=optimizer, train_evaluator=train_evaluator, val_evaluator=val_evaluator,
                         callback=callback, device=device)
        self.tokenizer = tokenizer
        self.grad_accum_steps = grad_accum_steps if grad_accum_steps and grad_accum_steps > 0 else 1
        self._step_in_accum = 0
    def on_train_start(self):
        self.model.train()

    def on_epoch_start(self):
        self.model.train()

    @torch.enable_grad()
    def batch_update(
        self,
        batch_idx: int,
        batch: Dict[str, Any],
        total_num_batches: Optional[int] = None,
        global_step: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Required by abstract base Trainer. Performs one optimization micro-step (or part of an accumulation step).
        Expects batch has 'input_ids', 'attention_mask', 'labels' (int64).
        Returns metrics dict for the TrainBatchOutputEvaluator.
        """
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # gradient accumulation
        # loss_to_backprop = loss / self.grad_accum
        loss_to_backprop = loss / self.grad_accum_steps
        loss_to_backprop.backward()
        self._step_in_accum += 1

        # stepped = False
        # if self._step_in_accum >= self.grad_accum:
        #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        #     self.optimizer.step()
        #     self.optimizer.zero_grad(set_to_none=True)
        #     self._step_in_accum = 0
        #     stepped = True

        # metrics = {"train loss": float(loss.detach().item())}

        # return metrics
        if self._step_in_accum >= self.grad_accum_steps:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._step_in_accum = 0

        return {"train loss": float(loss.detach().item())}

    def _forward_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        outputs = self.model(
            input_ids=batch["input_ids"].to(self.device),
            attention_mask=batch["attention_mask"].to(self.device),
            labels=batch["labels"].to(self.device),
        )
        # transformers causal LM returns loss when labels provided
        return outputs.loss

    def train_one_epoch(self, train_loader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        step = 0
        self.optimizer.zero_grad(set_to_none=True)

        for batch in train_loader:
            loss = self._forward_loss(batch) / self.grad_accum_steps
            loss.backward()
            step += 1
            total_loss += loss.item() * self.grad_accum_steps

            if step % self.grad_accum_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            # log to TrainEvaluator (only "train loss" here)
            self.train_evaluator.update({"train loss": loss.item() * self.grad_accum_steps})

        return total_loss / max(1, step)

    @torch.no_grad()
    def validate(self, val_loader) -> Optional[float]:
        if self.val_evaluator is None:
            return None
        self.model.eval()
        total = 0.0
        n = 0
        for batch in val_loader:
            loss = self._forward_loss(batch)
            total += float(loss.item())
            n += 1
        avg = total / max(1, n)
        self.val_evaluator.update({"train loss": avg})  # keep metric name consistent with plan
        return avg
