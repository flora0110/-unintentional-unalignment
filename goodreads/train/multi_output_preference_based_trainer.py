from common_dpo.trainers.dpo_trainer import DPOTrainer

class MultiOutputPreferenceBasedTrainer(DPOTrainer):
    def __init__(self, accelerator, config, model, tokenizer, dataset, optimizer_name="rmsprop",
                 assistant_token="<|assistant|>", ref_model=None, prompt_end_positions_func=None):
        self.trainer = DPOTrainer(
            accelerator=accelerator,
            config=config,                 # DPOConfig
            model=model,
            dataset=dataset,               # 需提前把 dataset 做成 (input_ids_w/l, attention_mask_w/l)
            ref_model=ref_model,
            tokenizer=tokenizer,
            assistant_token=assistant_token,
            optimizer_name=optimizer_name,
            prompt_end_positions_func=prompt_end_positions_func,
        )

    def train_one_epoch(self):
        for batch in self.trainer.dataloader:
            # 你的 collate_fn 應該已經做出下列四個張量
            stats = self.trainer.step(
                input_ids_w=batch["input_ids_w"],
                attention_mask_w=batch["attention_mask_w"],
                input_ids_l=batch["input_ids_l"],
                attention_mask_l=batch["attention_mask_l"],
            )
            self.trainer.log_stats(stats)