import logging
import math
from typing import List

import torch
from tabulate import tabulate

from common.train.callbacks import Callback
from common.train.trainer import Trainer
from common.utils.trie_utils import build_item_trie_and_first_token_set

class SequenceResp0LogitsAndProbsTrackerCallback(Callback):

    

    def __init__(self, tokenizer, logger: logging.Logger, num_inputs_to_log_logit_change_for: int = 1, top_tokens_to_log: int = 10, top_items_to_log: int = 20,
                 epoch_log_interval: int = 1, log_after_first_epoch: bool = True, track_finegrained_token_metrics: bool = False):
        self.tokenizer = tokenizer
        self.logger = logger
        self.num_inputs_to_log_logit_change_for = num_inputs_to_log_logit_change_for
        self.top_tokens_to_log = top_tokens_to_log
        self.top_items_to_log = top_items_to_log
        self.epoch_log_interval = epoch_log_interval
        self.log_after_first_epoch = log_after_first_epoch
        self.track_finegrained_token_metrics = track_finegrained_token_metrics

        # ---- id2name mapping ----
        with open(id2name_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # keys might be strings; normalize to int
        self.id2name: Dict[int, str] = {int(k): v for k, v in raw.items()}

        # ---- 建立 Trie 與預先 tokenized 的 item 序列（作為約束候選集）----
        self.item_trie, self.item_first_tokens = build_item_trie_and_first_token_set(
            {str(k): v for k, v in self.id2name.items()},  # 這個 helper 接受 str key
            self.tokenizer, add_leading_space=self.add_leading_space
        )
        # 預 tokenize：item_id -> List[int]
        self.item_tokenseqs: Dict[int, List[int]] = {}
        for iid, name in self.id2name.items():
            text = (" " + name) if self.add_leading_space else name
            toks = self.tokenizer.encode(text, add_special_tokens=False)[: self.max_item_len]
            if len(toks) == 0:
                continue
            self.item_tokenseqs[iid] = toks

        # 建立一個 item_id 與序列長度的張量，後續可向量化
        self._item_ids_sorted = sorted(self.item_tokenseqs.keys())
        self._item_tokenseqs_padded, self._item_attn_mask = self.__pad_item_tokenseqs(
            [self.item_tokenseqs[iid] for iid in self._item_ids_sorted]
        )  # -> LongTensor [N_items, L_item], Byte/Bool [N_items, L_item]
        # 快速查 id 索引
        self._itemid2col = {iid: j for j, iid in enumerate(self._item_ids_sorted)}


        # Initial quantities
        self.initial_logprobs = None
        self.initial_preferred_logprobs = None
        self.initial_dispreferred_logprobs = None
        self.initial_preferred_logprobs_mean = None
        self.initial_hidden_representations = None

        # Counters
        self.prev_epoch_logged_due_to_preferred_prob_decrease = -math.inf
        self.num_steps_preferred_prob_decreased = 0
        self.num_steps_train_loss_increase_when_preferred_prob_decreased = 0

        # Previous step quantities
        self.prev_epoch_logprobs = None
        self.prev_epoch_loss = None
        self.prev_epoch_token_unembeddings = None
        self.prev_epoch_hidden_representations = None

        # Aggregate quantities
        self.min_pref_logprob = None
        self.min_pref_logprob_smaller_than_init = False

        # Per example/step quantities
        self.per_step_per_example_is_pref_logprob_smaller_than_init = []
        self.per_step_per_example_did_preferred_prob_decrease = []
        self.per_step_did_train_loss_decrease = []
        self.per_step_per_example_preferred_token_logprob_change = []
        self.per_step_per_example_dispreferred_token_logprob_change = []

        self.until_step_per_example_preferred_token_logprob_increase_rank = []
        self.until_step_per_example_preferred_token_prob_increase_rank = []
        self.until_step_per_example_top_logprob_increase_token_ids = []
        self.until_step_per_example_top_logprob_increase_values = []
        self.until_step_per_example_top_prob_increase_token_ids = []
        self.until_step_per_example_top_prob_increase_values = []

        # ===== Sequence-level tracking (items) =====
        # Initial (ref at first seen), first-step policy (first time we see policy), latest/current policy values
        self.seq_init_ref_chosen = None       # Tensor[B]
        self.seq_init_ref_rejected = None
        self.seq_first_pol_chosen = None      # Tensor[B]
        self.seq_first_pol_rejected = None

        self.seq_curr_pol_chosen = None       # Tensor[B]
        self.seq_curr_pol_rejected = None
        self.seq_curr_item_ids_chosen = None  # LongTensor[B]
        self.seq_curr_item_ids_rejected = None

        self._seen_first_seq_step = False

    

    @torch.no_grad()
    def on_train_batch_end(self, trainer: Trainer, batch_num: int, batch_output, metric_values):
        # output_logprobs = batch_output["output logprobs"]
        # input_ids = batch_output["input ids"]
        # preferred_output_ids = batch_output["preferred output ids"]
        # dispreferred_output_ids = batch_output["dispreferred output ids"]
        # train_loss = batch_output["train loss"]
        # unembedding_weights = batch_output["unembedding weights"]
        # hidden_representations = batch_output["hidden representations"]

        # ========= Token-level (resp0) =========
        output_logprobs = batch_output["resp0_logprobs"]                 # [B, V]
        
        pol_logp_chosen = batch_output["seq/pol_logp_chosen"]      # [B]
        pol_logp_rejected = batch_output["seq/pol_logp_rejected"]    # [B]


        input_ids = batch_output["prompt_input_ids"]                     # [B, Lp]
        preferred_output_ids = batch_output["resp0_pref_ids"]            # [B]
        dispreferred_output_ids = batch_output["resp0_disp_ids"]         # [B]
        train_loss = batch_output["train loss"]
        unembedding_weights = batch_output["unembedding weights"]
        hidden_representations = batch_output["resp0_hidden"]            # [B, H]


        if self.initial_logprobs is None:
            # self.initial_logprobs = output_logprobs
            # self.initial_preferred_logprobs = output_logprobs[torch.arange(output_logprobs.size(0)), preferred_output_ids]
            # self.initial_dispreferred_logprobs = output_logprobs[torch.arange(output_logprobs.size(0)), dispreferred_output_ids]
            # self.initial_preferred_logprobs_mean = output_logprobs[torch.arange(output_logprobs.size(0)), preferred_output_ids].mean().item()
            # self.initial_hidden_representations = hidden_representations

            self.initial_logprobs = output_logprobs
            self.initial_preferred_logprobs = pol_logp_chosen
            self.initial_dispreferred_logprobs = pol_logp_rejected
            self.initial_preferred_logprobs_mean = pol_logp_chosen.mean().item()
            self.initial_hidden_representations = hidden_representations
        # else:
        #     logged = self.__log_if_preferred_token_prob_in_curr_step_decreased(trainer.epoch, output_logprobs, input_ids, preferred_output_ids,
        #                                                                        dispreferred_output_ids, train_loss, hidden_representations)

        #     should_log_token_logit_and_prob_stats = (self.epoch_log_interval > 0 and
        #                                              ((trainer.epoch + 1) % self.epoch_log_interval == 0 or trainer.epoch == 1))
        #     if should_log_token_logit_and_prob_stats and not logged:
        #         self.__log_token_logit_and_prob_stats_for_examples(trainer.epoch, output_logprobs, input_ids,
        #                                                            preferred_output_ids, dispreferred_output_ids, hidden_representations)
        #     if self.track_finegrained_token_metrics:
        #         self.__update_per_example_and_step_quantities(output_logprobs, preferred_output_ids, dispreferred_output_ids, train_loss)

        # self.__update_prev_epoch_and_aggregate_quantities(output_logprobs, preferred_output_ids, train_loss,
                                                        #   unembedding_weights, hidden_representations)
        else:
            # ---- 用序列級的 chosen/rejected logp 來偵測「偏好序列機率下降」 ----
            logged = self.__log_if_preferred_seq_prob_in_curr_step_decreased(
                epoch=trainer.epoch,
                pol_logp_chosen=pol_logp_chosen,           # [B]
                train_loss=train_loss,
            )

            # ---- token top-k 的表格仍照舊（for resp0 tokens）----
            should_log_token_logit_and_prob_stats = (
                self.epoch_log_interval > 0 and
                ((trainer.epoch + 1) % self.epoch_log_interval == 0 or trainer.epoch == 1)
            )
            if should_log_token_logit_and_prob_stats and not logged:
                self.__log_token_logit_and_prob_stats_for_examples(
                    trainer.epoch, output_logprobs, input_ids,
                    preferred_output_ids, dispreferred_output_ids, hidden_representations
                )

            # 如需逐步序列級度量（非必須）：把 per-step 的 chosen/rejected Δlogp 存起來
            if self.track_finegrained_token_metrics:
                self.__update_seq_per_example_and_step_quantities(
                    pol_logp_chosen=pol_logp_chosen,         # [B]
                    pol_logp_rejected=pol_logp_rejected,     # [B]
                    train_loss=train_loss
            )
        self.__update_prev_epoch_and_aggregate_quantities_seqaware(
            output_logprobs=output_logprobs,               # [B, V] 供 token top-k
            pol_logp_chosen=pol_logp_chosen,               # [B]    供 seq-level 聚合
            train_loss=train_loss,
            unembedding_weights=unembedding_weights,
            hidden_representations=hidden_representations
        )

        # ========= 新增：seq-level（Trie 受限候選，teacher forcing）=========
        # 使用同一批的 prompt（與你 token-level 計算一致）
        prompt_ids: torch.Tensor = batch_output["prompt_input_ids"].to(self.device)  # [B, Lp]
        if "prompt_attn_mask" in batch_output:
            prompt_attn = batch_output["prompt_attn_mask"].to(self.device)
        else:
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            prompt_attn = (prompt_ids != pad_id).to(self.device)

        # 候選過濾（可選）：用 policy 第一階段 token 分佈取 top-K 首 token；同時也可把 reference 的 top-K 併入聯集
        # 若你手上有當前步驟的「下一 token」logprob（與 token-level 相同），可用它來挑；否則直接評分全體候選
        candidate_cols = None
        if "resp0_logprobs" in batch_output:
            pol_first = batch_output["resp0_logprobs"].to(self.device)          # [B, V] (policy)
            # 若你也存有 reference 的第一步分佈，可合併；否則只用 policy 的
            candidate_cols = self.__candidate_items_by_top1st_token(pol_first, K=max(200, self.top_items_to_log*10))

        # 分別計算 policy / reference 的 item 序列 logp（[B, N_items]）
        seqlogp_pol = self.__seq_logp_over_items(self.policy_model,   prompt_ids, prompt_attn, candidate_item_cols=candidate_cols)
        seqlogp_ref = self.__seq_logp_over_items(self.reference_model, prompt_ids, prompt_attn, candidate_item_cols=candidate_cols)

        # 輸出整個 batch 的「Overall Top ↑ / ↓」
        should_log_seq = (self.epoch_log_interval > 0 and
                        ((trainer.epoch + 1) % self.epoch_log_interval == 0 or trainer.epoch == 1))
        if should_log_seq:
            self.__log_overall_seqlevel_stats(trainer.epoch, seqlogp_pol, seqlogp_ref, title_prefix="Overall Seq-Level (Trie constrained)")

        # 若你也想在第一個步驟存 baseline，可在這裡做（選擇性）
        if not hasattr(self, "seq_init_ref_all"):
            self.seq_init_ref_all = seqlogp_ref.detach().clone()
            self.seq_first_pol_all = seqlogp_pol.detach().clone()
        self.seq_curr_pol_all = seqlogp_pol.detach().clone()
        self.seq_curr_ref_all = seqlogp_ref.detach().clone()
        # ========= Sequence-level (items) =========
            # pol_ch = batch_output["seq/pol_logp_chosen"]      # [B]
            # pol_rj = batch_output["seq/pol_logp_rejected"]    # [B]
            # ref_ch = batch_output["seq/ref_logp_chosen"]      # [B]
            # ref_rj = batch_output["seq/ref_logp_rejected"]    # [B]
            # ids_ch = batch_output["chosen_item_ids"]          # LongTensor[B]
            # ids_rj = batch_output["rejected_item_ids"]        # LongTensor[B]

            # if self.seq_init_ref_chosen is None:
            #     # set initial (ref) baseline
            #     self.seq_init_ref_chosen = ref_ch.clone().detach()
            #     self.seq_init_ref_rejected = ref_rj.clone().detach()
            #     # also capture first-step policy (after first update)
            #     self.seq_first_pol_chosen = pol_ch.clone().detach()
            #     self.seq_first_pol_rejected = pol_rj.clone().detach()
            #     self._seen_first_seq_step = True

            # # always keep current/latest
            # self.seq_curr_pol_chosen = pol_ch.clone().detach()
            # self.seq_curr_pol_rejected = pol_rj.clone().detach()
            # self.seq_curr_item_ids_chosen = ids_ch.clone().detach()
            # self.seq_curr_item_ids_rejected = ids_rj.clone().detach()

    # @torch.no_grad()
    # def on_train_end(self, trainer: Trainer, fit_output):
    #     """
    #     At training end, log top-k items with largest increases/decreases in sequence logprob (chosen/rejected),
    #     including current value, first-step value, initial(ref) value, and deltas.
    #     """
    #     if self.seq_curr_pol_chosen is None or self.seq_init_ref_chosen is None:
    #         return  # no sequence info observed

    #     # ---- chosen ----
    #     self.__log_topk_items_table(
    #         title="FINAL Top-k CHOSEN items by logprob INCREASE (curr - init_ref)",
    #         curr=self.seq_curr_pol_chosen,
    #         first=self.seq_first_pol_chosen,
    #         init_ref=self.seq_init_ref_chosen,
    #         item_ids=self.seq_curr_item_ids_chosen,
    #         largest=True,
    #     )
    #     self.__log_topk_items_table(
    #         title="FINAL Top-k CHOSEN items by logprob DECREASE (curr - init_ref)",
    #         curr=self.seq_curr_pol_chosen,
    #         first=self.seq_first_pol_chosen,
    #         init_ref=self.seq_init_ref_chosen,
    #         item_ids=self.seq_curr_item_ids_chosen,
    #         largest=False,
    #     )

    #     # ---- rejected ----
    #     self.__log_topk_items_table(
    #         title="FINAL Top-k REJECTED items by logprob INCREASE (curr - init_ref)",
    #         curr=self.seq_curr_pol_rejected,
    #         first=self.seq_first_pol_rejected,
    #         init_ref=self.seq_init_ref_rejected,
    #         item_ids=self.seq_curr_item_ids_rejected,
    #         largest=True,
    #     )
    #     self.__log_topk_items_table(
    #         title="FINAL Top-k REJECTED items by logprob DECREASE (curr - init_ref)",
    #         curr=self.seq_curr_pol_rejected,
    #         first=self.seq_first_pol_rejected,
    #         init_ref=self.seq_init_ref_rejected,
    #         item_ids=self.seq_curr_item_ids_rejected,
    #         largest=False,
    #     )

    def __pad_item_tokenseqs(self, tokenseqs: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(tokenseqs) == 0:
            return torch.empty(0, 0, dtype=torch.long), torch.empty(0, 0, dtype=torch.bool)
        L = max(len(t) for t in tokenseqs)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        arr = torch.full((len(tokenseqs), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(tokenseqs), L), dtype=torch.bool)
        for i, t in enumerate(tokenseqs):
            n = len(t)
            arr[i, :n] = torch.tensor(t, dtype=torch.long)
            mask[i, :n] = True
        return arr.to(self.device), mask.to(self.device)

    @torch.no_grad()
    def __seq_logp_over_items(self, model: nn.Module,
                          prompt_input_ids: torch.Tensor,  # [B, Lp]
                          prompt_attn_mask: Optional[torch.Tensor] = None,
                          candidate_item_cols: Optional[torch.Tensor] = None
                          ) -> torch.Tensor:
        """
        回傳 shape [B, N_items] 的每個 item 的序列 logp（對齊 id2name 的候選集）。
        作法：teacher forcing，把每個 prompt 與所有候選 item_tokens 串接，收集 item token 位置的 log softmax 並加總。
        若 candidate_item_cols 非空，僅計算子集合（可用於先用第一步 top-K 先過濾）。
        """
        model.eval()

        B, Lp = prompt_input_ids.shape
        device = self.device
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        # 取要評分的 item 索引集合（列對應 self._item_ids_sorted 的 column 索引）
        if candidate_item_cols is None:
            cand_cols = torch.arange(self._item_tokenseqs_padded.size(0), device=device)
        else:
            cand_cols = candidate_item_cols.to(device)

        item_tok = self._item_tokenseqs_padded[cand_cols]        # [Nc, L_item]
        item_msk = self._item_attn_mask[cand_cols]               # [Nc, L_item]
        Nc, Litem = item_tok.shape

        # 我們採用「每個 batch 樣本 * 所有候選 item」的向量化展開：
        #   對於第 b 個樣本，複製 prompt  Nc 份；把 item_tok 拼接在每份 prompt 後面
        # 先把 prompt 展平到 [B*Nc, Lp]；item 複製到 [B*Nc, Litem]
        prompt_rep = prompt_input_ids.unsqueeze(1).expand(B, Nc, Lp).reshape(B * Nc, Lp)
        item_rep   = item_tok.unsqueeze(0).expand(B, Nc, Litem).reshape(B * Nc, Litem)

        inp = torch.cat([prompt_rep, item_rep], dim=1).to(device)              # [B*Nc, Lp+Litem]
        # 注意：若模型需要 attention_mask，則這裡也要拼接
        if prompt_attn_mask is None:
            pam = (prompt_rep != pad_id).to(device)
        else:
            pam = prompt_attn_mask.unsqueeze(1).expand(B, Nc, Lp).reshape(B * Nc, Lp).to(device)
        iam = item_msk.unsqueeze(0).expand(B, Nc, Litem).reshape(B * Nc, Litem).to(device)
        attn_mask = torch.cat([pam, iam], dim=1)                                # [B*Nc, Lp+Litem]

        # 前傳
        out = model(input_ids=inp, attention_mask=attn_mask, use_cache=False)
        # 取得所有位置的 logits；對 item token 的每一個位置 t，要拿位置 t 的 logits 去評分 token item_rep[:, t]
        # logits shape: [B*Nc, Lp+Litem, V]
        logits = out.logits[:, :-1, :]  # next-token logits；對齊輸入（shift）
        # item 的第一個 token 位置（在整串）是位置 Lp-1 的 next
        # 因此對 item 的 token 序列，我們需要 logits 的 slice：對應到輸入 index [Lp-1, Lp, ..., Lp+Litem-2] 的 next
        item_logits = logits[:, Lp-1: Lp-1 + Litem, :]                         # [B*Nc, Litem, V]
        logp = torch.log_softmax(item_logits, dim=-1)                          # [B*Nc, Litem, V]
        token_logp = torch.gather(logp, dim=-1, index=item_rep.unsqueeze(-1)).squeeze(-1)  # [B*Nc, Litem]

        # 僅加總有效的 item token（mask==True）
        token_logp = token_logp.masked_fill(~iam, 0.0)
        seq_logp = token_logp.sum(dim=1)                                       # [B*Nc]

        # reshape 回 [B, Nc]
        seq_logp = seq_logp.view(B, Nc)
        # 把 Nc 映回全體 N_items 的位置（沒有算的設為 -inf）
        N_all = self._item_tokenseqs_padded.size(0)
        full = torch.full((B, N_all), float('-inf'), device=device)
        full[:, cand_cols] = seq_logp
        return full  # [B, N_items]

    def __log_overall_seqlevel_stats(self, epoch: int,
                                 seqlogp_pol: torch.Tensor,  # [B, N_items]
                                 seqlogp_ref: torch.Tensor,  # [B, N_items]
                                 title_prefix: str = "Overall Seq-Level"):
        # 跨樣本平均（忽略 -inf）
        def masked_mean(x: torch.Tensor) -> torch.Tensor:
            m = torch.isfinite(x).float()
            num = (x * m).sum(dim=0)
            den = m.sum(dim=0).clamp_min(1.0)
            return num / den

        lp_pol = masked_mean(seqlogp_pol)  # [N_items]
        lp_ref = masked_mean(seqlogp_ref)  # [N_items]
        delta = lp_pol - lp_ref
        p_pol = torch.exp(lp_pol)
        p_ref = torch.exp(lp_ref)
        dp = p_pol - p_ref

        k = min(self.top_items_to_log, delta.numel())
        top_up   = torch.topk(delta, k=k, largest=True)
        top_down = torch.topk(delta, k=k, largest=False)
        top_p_up   = torch.topk(dp, k=k, largest=True)
        top_p_down = torch.topk(dp, k=k, largest=False)

        def rows_from_indices(idx_tensor: torch.Tensor, header: str):
            rows = [header]
            for j in idx_tensor.tolist():
                iid = self._item_ids_sorted[j]
                name = self.id2name.get(iid, f"<{iid}>")
                rows.append(f"(id={iid}) {name} | p_ref={p_ref[j].item():.6e} | p_pol={p_pol[j].item():.6e} | "
                            f"delta_logp={delta[j].item():.6f}")
            return rows

        table_rows = []
        table_rows.append(rows_from_indices(top_up.indices,   "Overall Top Logprob Increase"))
        table_rows.append(rows_from_indices(top_down.indices, "Overall Top Logprob Decrease"))
        table_rows.append(rows_from_indices(top_p_up.indices,   "Overall Top Prob Increase"))
        table_rows.append(rows_from_indices(top_p_down.indices, "Overall Top Prob Decrease"))

        self.__log_table(epoch, f"{title_prefix} (k={k})", row_values=table_rows, additional_info=None)

    def __log_if_preferred_seq_prob_in_curr_step_decreased(self, epoch: int, pol_logp_chosen: torch.Tensor, train_loss: float):
        """
        用序列級 chosen 的 logp 平均值判斷是否較「上一輪」下降；若是且到達 log 週期，輸出提示。
        """
        # 上一步的序列級「上一輪平均」若未就緒（第一步），就不觸發
        if self.prev_epoch_logprobs is None:
            return False

        mean_curr = pol_logp_chosen.mean()
        # 用前一輪的「序列級 chosen 平均」：我們把它存在 self.prev_epoch_seq_pref_mean
        mean_prev = self.prev_epoch_seq_pref_mean if hasattr(self, "prev_epoch_seq_pref_mean") else mean_curr

        prob_decreased = (mean_curr - mean_prev) < 0
        if (not prob_decreased) or (self.prev_epoch_logged_due_to_preferred_prob_decrease + self.epoch_log_interval > epoch):
            # 沒下降或未到 log 間隔，就不記錄
            self.prev_epoch_seq_pref_mean = mean_curr
            return False

        # 記錄一次
        self.prev_epoch_logged_due_to_preferred_prob_decrease = epoch
        self.num_steps_preferred_prob_decreased += 1
        train_loss_increased = train_loss > (self.prev_epoch_loss if self.prev_epoch_loss is not None else train_loss)
        if train_loss_increased:
            self.num_steps_train_loss_increase_when_preferred_prob_decreased += 1

        if self.epoch_log_interval <= 0:
            self.prev_epoch_seq_pref_mean = mean_curr
            return False

        init_mean = self.initial_preferred_logprobs_mean  # 你在 __init__ 時存的 seq-level 初始（來自第一步 pol_chosen 平均）
        self.logger.info(
            f"\n**********************************************************\n"
            f"Epoch: {epoch}: Mean preferred **sequence** log probability decreased from previous step: "
            f"{mean_curr.item():.6f} - {mean_prev.item():.6f} "
            f"(diff: {(mean_curr - mean_prev).item():.6f}, init: {init_mean:.6f})\n"
            f"Train loss after step {train_loss:.6f} (increased vs prev? {train_loss_increased})\n"
            f"Total #steps where preferred **sequence** prob decreased: {self.num_steps_preferred_prob_decreased}\n"
            f"**********************************************************"
        )
        self.prev_epoch_seq_pref_mean = mean_curr
        return True

    def __update_prev_epoch_and_aggregate_quantities_seqaware(
        self,
        output_logprobs: torch.Tensor,         # [B, V] 仍保留給 token top-k 使用
        pol_logp_chosen: torch.Tensor,         # [B]    用來做 seq-level 聚合
        train_loss: float,
        unembedding_weights: torch.Tensor,
        hidden_representations: torch.Tensor
    ):
        # ---- for token analytics (與原本一致) ----
        self.prev_epoch_loss = train_loss
        self.prev_epoch_logprobs = output_logprobs
        self.prev_epoch_token_unembeddings = unembedding_weights
        self.prev_epoch_hidden_representations = hidden_representations

        # ---- for sequence-level aggregates ----
        curr_pref_seq_mean = pol_logp_chosen.mean().item()
        if self.min_pref_logprob is None:
            self.min_pref_logprob = curr_pref_seq_mean
        else:
            self.min_pref_logprob = min(self.min_pref_logprob, curr_pref_seq_mean)

        self.min_pref_logprob_smaller_than_init = (
            self.min_pref_logprob_smaller_than_init or
            (self.min_pref_logprob < self.initial_preferred_logprobs_mean)
        )

        # 存下本輪的序列級「平均」以供下一步比較
        self.prev_epoch_seq_pref_mean = pol_logp_chosen.mean()


    # old token-level methods below
    

    def __update_prev_epoch_and_aggregate_quantities(self, output_logprobs, preferred_output_ids, train_loss, unembedding_weights,
                                                     hidden_representations):
        self.prev_epoch_loss = train_loss
        self.prev_epoch_logprobs = output_logprobs

        curr_pref_log_prob = output_logprobs[torch.arange(output_logprobs.size(0)), preferred_output_ids].mean().item()
        self.min_pref_logprob = min(self.min_pref_logprob, curr_pref_log_prob) if self.min_pref_logprob is not None else curr_pref_log_prob
        self.min_pref_logprob_smaller_than_init = (self.min_pref_logprob_smaller_than_init or
                                                   self.min_pref_logprob < self.initial_preferred_logprobs_mean)

        self.prev_epoch_token_unembeddings = unembedding_weights
        self.prev_epoch_hidden_representations = hidden_representations

    def __update_per_example_and_step_quantities(self, output_logprobs, preferred_output_ids, dispreferred_output_ids, train_loss):
        preferred_logprobs = output_logprobs[torch.arange(output_logprobs.size(0)), preferred_output_ids]
        dispreferred_logprobs = output_logprobs[torch.arange(output_logprobs.size(0)), dispreferred_output_ids]

        initial_preferred_logprobs = self.initial_logprobs[torch.arange(output_logprobs.size(0)), preferred_output_ids]
        self.per_step_per_example_is_pref_logprob_smaller_than_init.append((preferred_logprobs < initial_preferred_logprobs).cpu())
        initial_dispreferred_logprobs = self.initial_logprobs[torch.arange(output_logprobs.size(0)), dispreferred_output_ids]

        prev_epoch_preferred_logprobs = self.prev_epoch_logprobs[torch.arange(output_logprobs.size(0)), preferred_output_ids]
        self.per_step_per_example_did_preferred_prob_decrease.append((preferred_logprobs < prev_epoch_preferred_logprobs).cpu())
        self.per_step_per_example_preferred_token_logprob_change.append((preferred_logprobs - initial_preferred_logprobs).cpu())
        self.per_step_per_example_dispreferred_token_logprob_change.append((dispreferred_logprobs - initial_dispreferred_logprobs).cpu())
        self.per_step_did_train_loss_decrease.append(train_loss <= self.prev_epoch_loss)

        # Until step prob/log prob increase quantities
        overall_logprobs_change = output_logprobs - self.initial_logprobs
        overall_sorted_logprobs_change_indices = torch.argsort(overall_logprobs_change, dim=1, descending=True)
        overall_logprob_change_ranks_of_preferred_token = torch.where(overall_sorted_logprobs_change_indices == preferred_output_ids.view(-1, 1))[1]
        self.until_step_per_example_preferred_token_logprob_increase_rank.append(overall_logprob_change_ranks_of_preferred_token.cpu())

        overall_probs_change = torch.exp(output_logprobs) - torch.exp(self.initial_logprobs)
        overall_sorted_probs_change_indices = torch.argsort(overall_probs_change, dim=1, descending=True)
        overall_prob_change_ranks_of_preferred_token = torch.where(overall_sorted_probs_change_indices == preferred_output_ids.view(-1, 1))[1]
        self.until_step_per_example_preferred_token_prob_increase_rank.append(overall_prob_change_ranks_of_preferred_token.cpu())

        self.until_step_per_example_top_logprob_increase_token_ids.append(overall_sorted_logprobs_change_indices[:, :self.top_tokens_to_log].cpu())
        self.until_step_per_example_top_logprob_increase_values.append(torch.gather(overall_logprobs_change, dim=1,
                                                                                    index=overall_sorted_logprobs_change_indices[:,
                                                                                          :self.top_tokens_to_log]).cpu())
        self.until_step_per_example_top_prob_increase_token_ids.append(overall_sorted_probs_change_indices[:, :self.top_tokens_to_log].cpu())
        self.until_step_per_example_top_prob_increase_values.append(torch.gather(overall_probs_change, dim=1,
                                                                                 index=overall_sorted_probs_change_indices[:,
                                                                                       :self.top_tokens_to_log]).cpu())

    def __log_token_logit_and_prob_stats_for_examples(self, epoch: int, output_logprobs: torch.Tensor, input_ids: torch.Tensor,
                                                      preferred_output_ids: torch.Tensor, dispreferred_output_ids: torch.Tensor,
                                                      hidden_representations: torch.Tensor):
        for i in range(self.num_inputs_to_log_logit_change_for):
            self.__log_token_logit_and_prob_table(epoch, i, torch.exp(output_logprobs[i]), output_logprobs[i], input_ids[i],
                                                  preferred_output_ids[i], dispreferred_output_ids[i], hidden_representations[i])

    def __log_if_preferred_token_prob_in_curr_step_decreased(self, epoch: int, output_logprobs: torch.Tensor, input_ids: torch.Tensor,
                                                             preferred_output_ids: torch.Tensor, dispreferred_output_ids: torch.Tensor,
                                                             train_loss: float, hidden_representations: torch.Tensor):
        mean_preferred_log_probs = output_logprobs[torch.arange(output_logprobs.size(0)), preferred_output_ids].mean()
        mean_prev_epoch_preferred_logprobs = self.prev_epoch_logprobs[torch.arange(output_logprobs.size(0)), preferred_output_ids].mean()

        prob_decreased = (mean_preferred_log_probs - mean_prev_epoch_preferred_logprobs) < 0
        if not prob_decreased or self.prev_epoch_logged_due_to_preferred_prob_decrease + self.epoch_log_interval > epoch:
            return False

        self.prev_epoch_logged_due_to_preferred_prob_decrease = epoch
        self.num_steps_preferred_prob_decreased += 1
        train_loss_increased = train_loss > self.prev_epoch_loss
        if train_loss_increased:
            self.num_steps_train_loss_increase_when_preferred_prob_decreased += 1

        if self.epoch_log_interval <= 0:
            return False

        self.logger.info(f"\n**********************************************************\n"
                         f"Epoch: {epoch}: Mean preferred token log probability decreased from the previous step: "
                         f"{mean_preferred_log_probs.item():.6f} - {mean_prev_epoch_preferred_logprobs.item():.6f} "
                         f"(diff: {(mean_preferred_log_probs - mean_prev_epoch_preferred_logprobs).item():.6f}, "
                         f"init: {self.initial_logprobs[torch.arange(output_logprobs.size(0)), preferred_output_ids].mean().item():.6f})\n"
                         f"Train loss after step {train_loss:.6f} and before step {self.prev_epoch_loss:.6f} (increased? {train_loss_increased})\n"
                         f"Total number of steps in which preferred token probability decreased thus far: {self.num_steps_preferred_prob_decreased}\n"
                         f"**********************************************************")

        self.__log_token_logit_and_prob_stats_for_examples(epoch, output_logprobs, input_ids,
                                                           preferred_output_ids, dispreferred_output_ids, hidden_representations)
        return True

    def __log_token_logit_and_prob_table(self, epoch: int, example_index: int, output_probs: torch.Tensor,
                                         output_logprobs: torch.Tensor, input_ids: torch.Tensor,
                                         preferred_output_id: torch.Tensor, dispreferred_output_id: torch.Tensor,
                                         hidden_representation: torch.Tensor):
        input = self.tokenizer.decode(input_ids)
        preferred_output = self.tokenizer.decode(preferred_output_id)
        dispreferred_output = self.tokenizer.decode(dispreferred_output_id)
        table_title = (f"Top {self.top_tokens_to_log} token logit and probability statistics for input: '{input}' ,"
                       f" preferred output: '{preferred_output}' , dispreferred output: '{dispreferred_output}'")

        initial_probs = torch.exp(self.initial_logprobs[example_index])
        prev_epoch_probs = torch.exp(self.prev_epoch_logprobs[example_index])
        curr_step_logprobs_change = output_logprobs - self.prev_epoch_logprobs[example_index]
        curr_step_probs_change = output_probs - prev_epoch_probs
        overall_logprobs_change = output_logprobs - self.initial_logprobs[example_index]
        overall_probs_change = output_probs - initial_probs

        table_row_values = []
        self.__populate_row_values_with_curr_step_token_prob_changes(table_row_values, example_index, output_probs, output_logprobs, initial_probs,
                                                                     prev_epoch_probs, curr_step_probs_change, curr_step_logprobs_change)
        self.__populate_row_values_with_overall_token_prob_changes(table_row_values, example_index, output_probs, output_logprobs, initial_probs,
                                                                   overall_probs_change, overall_logprobs_change)
        additional_info = self.__populate_row_values_with_curr_step_representation_metrics(table_row_values, example_index, curr_step_logprobs_change,
                                                                                           curr_step_probs_change, preferred_output_id,
                                                                                           dispreferred_output_id, hidden_representation)

        self.__log_table(epoch, table_title, row_values=table_row_values, additional_info=additional_info)

    def __populate_row_values_with_curr_step_token_prob_changes(self, row_values, example_index, output_probs, output_logprobs, initial_probs,
                                                                prev_epoch_probs, curr_step_probs_change, curr_step_logprobs_change):
        curr_step_top_logprob_increase_token_ids = torch.topk(curr_step_logprobs_change, k=self.top_tokens_to_log, largest=True).indices
        curr_step_top_logprob_increase_tokens = self.tokenizer.convert_ids_to_tokens(curr_step_top_logprob_increase_token_ids.view(-1, 1))
        curr_step_top_logprob_increase_decoded_tokens = self.tokenizer.batch_decode(curr_step_top_logprob_increase_token_ids.view(-1, 1))
        curr_step_top_logprob_increase_cell_strings = [(f"{token} (dec: {repr(decoded_token)}): {new_logprob:.6f} - {prev_logprob:.6f} "
                                                        f"(diff: {change:.6f}, init: {init_logprob:.6f})")
                                                       for token, decoded_token, new_logprob, prev_logprob, init_logprob, change in
                                                       zip(curr_step_top_logprob_increase_tokens,
                                                           curr_step_top_logprob_increase_decoded_tokens,
                                                           output_logprobs[curr_step_top_logprob_increase_token_ids],
                                                           self.prev_epoch_logprobs[example_index][curr_step_top_logprob_increase_token_ids],
                                                           self.initial_logprobs[example_index][curr_step_top_logprob_increase_token_ids],
                                                           curr_step_logprobs_change[curr_step_top_logprob_increase_token_ids])]

        row_values.append(["Curr Step Top Logprob Increase"] + curr_step_top_logprob_increase_cell_strings)

        curr_step_top_logprob_decrease_token_ids = torch.topk(curr_step_logprobs_change, k=self.top_tokens_to_log, largest=False).indices
        curr_step_top_logprob_decrease_tokens = self.tokenizer.convert_ids_to_tokens(curr_step_top_logprob_decrease_token_ids.view(-1, 1))
        curr_step_top_logprob_decrease_decoded_tokens = self.tokenizer.batch_decode(curr_step_top_logprob_decrease_token_ids.view(-1, 1))
        curr_step_top_logprob_decrease_cell_strings = [(f"{token} (dec: {repr(decoded_token)}): {new_logprob:.6f} - {prev_logprob:.6f} "
                                                        f"(diff: {change:.6f}, init: {init_logprob:.6f})")
                                                       for token, decoded_token, new_logprob, prev_logprob, init_logprob, change in
                                                       zip(curr_step_top_logprob_decrease_tokens,
                                                           curr_step_top_logprob_decrease_decoded_tokens,
                                                           output_logprobs[curr_step_top_logprob_decrease_token_ids],
                                                           self.prev_epoch_logprobs[example_index][curr_step_top_logprob_decrease_token_ids],
                                                           self.initial_logprobs[example_index][curr_step_top_logprob_decrease_token_ids],
                                                           curr_step_logprobs_change[curr_step_top_logprob_decrease_token_ids])]

        row_values.append(["Curr Step Top Logprob Decrease"] + curr_step_top_logprob_decrease_cell_strings)

        curr_step_top_prob_increase_token_ids = torch.topk(curr_step_probs_change, k=self.top_tokens_to_log, largest=True).indices
        curr_step_top_prob_increase_tokens = self.tokenizer.convert_ids_to_tokens(curr_step_top_prob_increase_token_ids.view(-1, 1))
        curr_step_top_prob_increase_decoded_tokens = self.tokenizer.batch_decode(curr_step_top_prob_increase_token_ids.view(-1, 1))
        curr_step_top_prob_increase_cell_strings = [(f"{token} (dec: {repr(decoded_token)}): {new_prob:.6f} - {prev_prob:.6f} (diff: {change:.6f}, "
                                                     f"init: {init_prob:.6f})")
                                                    for token, decoded_token, new_prob, prev_prob, init_prob, change in
                                                    zip(curr_step_top_prob_increase_tokens,
                                                        curr_step_top_prob_increase_decoded_tokens,
                                                        output_probs[curr_step_top_prob_increase_token_ids],
                                                        prev_epoch_probs[curr_step_top_prob_increase_token_ids],
                                                        initial_probs[curr_step_top_prob_increase_token_ids],
                                                        curr_step_probs_change[curr_step_top_prob_increase_token_ids])]

        row_values.append(["Curr Step Top Prob Increase"] + curr_step_top_prob_increase_cell_strings)

        curr_step_top_prob_decrease_token_ids = torch.topk(curr_step_probs_change, k=self.top_tokens_to_log, largest=False).indices
        curr_step_top_prob_decrease_tokens = self.tokenizer.convert_ids_to_tokens(curr_step_top_prob_decrease_token_ids.view(-1, 1))
        curr_step_top_prob_decrease_decoded_tokens = self.tokenizer.batch_decode(curr_step_top_prob_decrease_token_ids.view(-1, 1))
        curr_step_top_prob_decrease_cell_strings = [
            f"{token} (dec: {repr(decoded_token)}): {new_prob:.6f} - {prev_prob:.6f} (diff: {change:.6f}, init: {init_prob:.6f})"
            for token, decoded_token, new_prob, prev_prob, init_prob, change in
            zip(curr_step_top_prob_decrease_tokens,
                curr_step_top_prob_decrease_decoded_tokens,
                output_probs[curr_step_top_prob_decrease_token_ids],
                prev_epoch_probs[curr_step_top_prob_decrease_token_ids],
                initial_probs[curr_step_top_prob_decrease_token_ids],
                curr_step_probs_change[curr_step_top_prob_decrease_token_ids])]

        row_values.append(["Curr Step Top Prob Decrease"] + curr_step_top_prob_decrease_cell_strings)

    def __populate_row_values_with_overall_token_prob_changes(self, row_values, example_index, output_probs, output_logprobs, initial_probs,
                                                              overall_probs_change, overall_logprobs_change):
        overall_top_logprob_increase_token_ids = torch.topk(overall_logprobs_change, k=self.top_tokens_to_log, largest=True).indices
        overall_top_logprob_increase_tokens = self.tokenizer.convert_ids_to_tokens(overall_top_logprob_increase_token_ids.view(-1, 1))
        overall_top_logprob_increase_decoded_tokens = self.tokenizer.batch_decode(overall_top_logprob_increase_token_ids.view(-1, 1))
        overall_top_logprob_increase_cell_strings = [
            f"{token} (dec: {repr(decoded_token)}): {new_logprob:.6f} - {init_logprob:.6f} (diff: {change:.6f})"
            for token, decoded_token, new_logprob, init_logprob, change in
            zip(overall_top_logprob_increase_tokens,
                overall_top_logprob_increase_decoded_tokens,
                output_logprobs[overall_top_logprob_increase_token_ids],
                self.initial_logprobs[example_index][overall_top_logprob_increase_token_ids],
                overall_logprobs_change[overall_top_logprob_increase_token_ids])]

        row_values.append(["Overall Top Logprob Increase"] + overall_top_logprob_increase_cell_strings)

        overall_top_logprob_decrease_token_ids = torch.topk(overall_logprobs_change, k=self.top_tokens_to_log, largest=False).indices
        overall_top_logprob_decrease_tokens = self.tokenizer.convert_ids_to_tokens(overall_top_logprob_decrease_token_ids.view(-1, 1))
        overall_top_logprob_decrease_decoded_tokens = self.tokenizer.batch_decode(overall_top_logprob_decrease_token_ids.view(-1, 1))
        overall_top_logprob_decrease_cell_strings = [
            f"{token} (dec: {repr(decoded_token)}): {new_logprob:.6f} - {init_logprob:.6f} (diff: {change:.6f})"
            for token, decoded_token, new_logprob, init_logprob, change in
            zip(overall_top_logprob_decrease_tokens,
                overall_top_logprob_decrease_decoded_tokens,
                output_logprobs[overall_top_logprob_decrease_token_ids],
                self.initial_logprobs[example_index][overall_top_logprob_decrease_token_ids],
                overall_logprobs_change[overall_top_logprob_decrease_token_ids])]

        row_values.append(["Overall Top Logprob Decrease"] + overall_top_logprob_decrease_cell_strings)

        overall_top_prob_increase_token_ids = torch.topk(overall_probs_change, k=self.top_tokens_to_log, largest=True).indices
        overall_top_prob_increase_tokens = self.tokenizer.convert_ids_to_tokens(overall_top_prob_increase_token_ids.view(-1, 1))
        overall_top_prob_increase_decoded_tokens = self.tokenizer.batch_decode(overall_top_prob_increase_token_ids.view(-1, 1))
        overall_top_prob_increase_cell_strings = [f"{token} (dec: {repr(decoded_token)}): {new_prob:.6f} - {init_prob:.6f} (diff: {change:.6f})"
                                                  for token, decoded_token, new_prob, init_prob, change in
                                                  zip(overall_top_prob_increase_tokens,
                                                      overall_top_prob_increase_decoded_tokens,
                                                      output_probs[overall_top_prob_increase_token_ids],
                                                      initial_probs[overall_top_prob_increase_token_ids],
                                                      overall_probs_change[overall_top_prob_increase_token_ids])]

        row_values.append(["Overall Top Prob Increase"] + overall_top_prob_increase_cell_strings)

        overall_top_prob_decrease_token_ids = torch.topk(overall_probs_change, k=self.top_tokens_to_log, largest=False).indices
        overall_top_prob_decrease_tokens = self.tokenizer.convert_ids_to_tokens(overall_top_prob_decrease_token_ids.view(-1, 1))
        overall_top_prob_decrease_decoded_tokens = self.tokenizer.batch_decode(overall_top_prob_decrease_token_ids.view(-1, 1))
        overall_top_prob_decrease_cell_strings = [f"{token} (dec: {repr(decoded_token)}): {new_prob:.6f} - {init_prob:.6f} (diff: {change:.6f})"
                                                  for token, decoded_token, new_prob, init_prob, change in
                                                  zip(overall_top_prob_decrease_tokens,
                                                      overall_top_prob_decrease_decoded_tokens,
                                                      output_probs[overall_top_prob_decrease_token_ids],
                                                      initial_probs[overall_top_prob_decrease_token_ids],
                                                      overall_probs_change[overall_top_prob_decrease_token_ids])]

        row_values.append(["Overall Top Prob Decrease"] + overall_top_prob_decrease_cell_strings)

        curr_top_prob_token_ids = torch.topk(output_probs, k=self.top_tokens_to_log, largest=True).indices
        curr_top_prob_tokens = self.tokenizer.convert_ids_to_tokens(curr_top_prob_token_ids.view(-1, 1))
        curr_top_prob_decoded_tokens = self.tokenizer.batch_decode(curr_top_prob_token_ids.view(-1, 1))
        curr_top_prob_tokens_cell_strings = [f"{token} (dec: {repr(decoded_token)}): {curr_prob:.6f} (initial: {init_prob:.6f})"
                                             for token, decoded_token, curr_prob, init_prob in
                                             zip(curr_top_prob_tokens,
                                                 curr_top_prob_decoded_tokens,
                                                 output_probs[curr_top_prob_token_ids],
                                                 initial_probs[curr_top_prob_token_ids])]
        curr_top_logprob_tokens_cell_strings = [(f"{token} (dec: {repr(decoded_token)}): {torch.log(curr_prob):.6f}"
                                                 f" (initial: {torch.log(init_prob):.6f})")
                                                for token, decoded_token, curr_prob, init_prob in
                                                zip(curr_top_prob_tokens,
                                                    curr_top_prob_decoded_tokens,
                                                    output_probs[curr_top_prob_token_ids],
                                                    initial_probs[curr_top_prob_token_ids])]

        row_values.append(["Curr Top Prob"] + curr_top_prob_tokens_cell_strings)
        row_values.append(["Curr Top Logprob"] + curr_top_logprob_tokens_cell_strings)

        initial_top_prob_token_ids = torch.topk(initial_probs, k=self.top_tokens_to_log, largest=True).indices
        initial_top_prob_tokens = self.tokenizer.convert_ids_to_tokens(initial_top_prob_token_ids.view(-1, 1))
        initial_top_prob_decoded_tokens = self.tokenizer.batch_decode(initial_top_prob_token_ids.view(-1, 1))
        initial_top_prob_tokens_cell_strings = [f"{token} (dec: {repr(decoded_token)}): {init_prob:.6f}"
                                                for token, decoded_token, init_prob in
                                                zip(initial_top_prob_tokens,
                                                    initial_top_prob_decoded_tokens,
                                                    initial_probs[initial_top_prob_token_ids])]
        initial_top_logprob_tokens_cell_strings = [f"{token} (dec: {repr(decoded_token)}): {torch.log(init_prob):.6f}"
                                                   for token, decoded_token, init_prob in
                                                   zip(initial_top_prob_tokens,
                                                       initial_top_prob_decoded_tokens,
                                                       initial_probs[initial_top_prob_token_ids])]

        row_values.append(["Initial Top Prob"] + initial_top_prob_tokens_cell_strings)
        row_values.append(["Initial Top Logprob"] + initial_top_logprob_tokens_cell_strings)

    def __populate_row_values_with_curr_step_representation_metrics(self, row_values, example_index, curr_step_logprobs_change,
                                                                    curr_step_probs_change, preferred_output_id, dispreferred_output_id,
                                                                    hidden_representation):
        preferred_token_embedding = self.prev_epoch_token_unembeddings[preferred_output_id]
        dispreferred_token_embedding = self.prev_epoch_token_unembeddings[dispreferred_output_id]
        unembedding_inner_prods = torch.matmul(self.prev_epoch_token_unembeddings, preferred_token_embedding - dispreferred_token_embedding)

        top_logprob_increase_token_ids = torch.topk(curr_step_logprobs_change, k=self.top_tokens_to_log, largest=True).indices.cpu()
        top_logprob_tokens = self.tokenizer.convert_ids_to_tokens(top_logprob_increase_token_ids.view(-1, 1))
        top_logprob_decoded_tokens = self.tokenizer.batch_decode(top_logprob_increase_token_ids.view(-1, 1))
        top_logprob_inner_prod_cell_strings = [f"{token} (dec: {repr(decoded_token)}): {inner_prod:.6f}"
                                               for token, decoded_token, inner_prod in
                                               zip(top_logprob_tokens,
                                                   top_logprob_decoded_tokens,
                                                   unembedding_inner_prods[top_logprob_increase_token_ids])]

        row_values.append(["<W_y , W_p - W_d> For Curr Step Top Logprob Increase"] + top_logprob_inner_prod_cell_strings)

        top_prob_increase_token_ids = torch.topk(curr_step_probs_change, k=self.top_tokens_to_log, largest=True).indices.cpu()
        top_prob_tokens = self.tokenizer.convert_ids_to_tokens(top_prob_increase_token_ids.view(-1, 1))
        top_prob_decoded_tokens = self.tokenizer.batch_decode(top_prob_increase_token_ids.view(-1, 1))
        top_prob_inner_prod_cell_strings = [f"{token} (dec: {repr(decoded_token)}): {inner_prod:.6f}"
                                            for token, decoded_token, inner_prod in
                                            zip(top_prob_tokens,
                                                top_prob_decoded_tokens,
                                                unembedding_inner_prods[top_prob_increase_token_ids])]

        row_values.append(["<W_y , W_p - W_d> For Curr Step Top Prob Increase"] + top_prob_inner_prod_cell_strings)

        top_inner_prod_token_ids = torch.topk(unembedding_inner_prods, self.top_tokens_to_log, largest=True).indices.cpu()
        top_inner_prod_tokens = self.tokenizer.convert_ids_to_tokens(top_inner_prod_token_ids.view(-1, 1))
        top_inner_prod_decoded_tokens = self.tokenizer.batch_decode(top_inner_prod_token_ids.view(-1, 1))
        top_inner_prod_cell_strings = [f"{token} (dec: {repr(decoded_token)}): {inner_prod:.6f}"
                                       for token, decoded_token, inner_prod in
                                       zip(top_inner_prod_tokens,
                                           top_inner_prod_decoded_tokens,
                                           unembedding_inner_prods[top_inner_prod_token_ids])]

        row_values.append(["Curr Step Top <W_y , W_p - W_d>"] + top_inner_prod_cell_strings)

        bottom_inner_prod_token_ids = torch.topk(unembedding_inner_prods, self.top_tokens_to_log, largest=False).indices.cpu()
        bottom_inner_prod_tokens = self.tokenizer.convert_ids_to_tokens(bottom_inner_prod_token_ids.view(-1, 1))
        bottom_inner_prod_decoded_tokens = self.tokenizer.batch_decode(bottom_inner_prod_token_ids.view(-1, 1))
        bottom_inner_prod_cell_strings = [f"{token} (dec: {repr(decoded_token)}): {inner_prod:.6f}"
                                          for token, decoded_token, inner_prod in
                                          zip(bottom_inner_prod_tokens,
                                              bottom_inner_prod_decoded_tokens,
                                              unembedding_inner_prods[bottom_inner_prod_token_ids])]

        row_values.append(["Curr Step Bottom <W_y , W_p - W_d>"] + bottom_inner_prod_cell_strings)

        prev_epoch_hidden_representation = self.prev_epoch_hidden_representations[example_index]
        hidden_repr_inner_prod = torch.inner(prev_epoch_hidden_representation,
                                             (preferred_token_embedding - dispreferred_token_embedding).to(prev_epoch_hidden_representation.device))
        hidden_repr_sq_norm = (prev_epoch_hidden_representation ** 2).sum()
        hidden_repr_sq_dist_from_init = ((prev_epoch_hidden_representation - self.initial_hidden_representations[example_index]) ** 2).sum()
        hidden_repr_curr_step_sq_dist = ((prev_epoch_hidden_representation - hidden_representation) ** 2).sum()

        num_largest_inner_prod_and_logprob_change = self.__compute_intersection_size(top_logprob_increase_token_ids,
                                                                                     top_inner_prod_token_ids)
        num_largest_inner_prod_and_prob_change = self.__compute_intersection_size(top_prob_increase_token_ids,
                                                                                  top_inner_prod_token_ids)

        additional_info = (f"Additional representations info:\n"
                           f"<W_p , W_p - W_d>: {unembedding_inner_prods[preferred_output_id].item():.6f}  "
                           f", <W_d , W_p - W_d>: {unembedding_inner_prods[dispreferred_output_id].item():.6f}  "
                           f", <W_p ,W_d>: {torch.inner(preferred_token_embedding, dispreferred_token_embedding).item():.6f}  "
                           f", ||W_p||^2: {(preferred_token_embedding ** 2).sum().item():.6f}  "
                           f", ||W_d||^2 {(dispreferred_token_embedding ** 2).sum().item():.6f}  "
                           f", <h , W_p - W_d>: {hidden_repr_inner_prod.item():.6f}  "
                           f", <h , W_p>: {torch.inner(prev_epoch_hidden_representation, preferred_token_embedding.to(prev_epoch_hidden_representation.device)).item():.6f}  "
                           f", <h , W_d>: {torch.inner(prev_epoch_hidden_representation, dispreferred_token_embedding.to(prev_epoch_hidden_representation.device)).item():.6f}  "
                           f", ||h||^2: {hidden_repr_sq_norm.item()}  "
                           f", ||h - h_init||^2: {hidden_repr_sq_dist_from_init.item()}  "
                           f", ||h^+ - h||^2: {hidden_repr_curr_step_sq_dist.item()}\n"
                           f"Num tokens with highest <W_y , W_p - W_d> and largest curr step logprob change:"
                           f" {num_largest_inner_prod_and_logprob_change} / {self.top_tokens_to_log}\n"
                           f"Num tokens with highest <W_y , W_p - W_d> and largest curr step prob change:"
                           f" {num_largest_inner_prod_and_prob_change} / {self.top_tokens_to_log}")
        return additional_info

    def __compute_intersection_size(self, first_indices_tensor: torch.Tensor, second_indices_tensor: torch.Tensor):
        set1 = set(first_indices_tensor.tolist())
        set2 = set(second_indices_tensor.tolist())
        common_indices = set1.intersection(set2)
        return len(common_indices)

    def __log_table(self, epoch: int, title: str, row_values: List[List[str]], additional_info: str = None):
        log_str = (f"===========================================================================\n"
                   f"Epoch: {epoch}, {title}\n{tabulate(row_values, tablefmt='pretty')}\n"
                   f"===========================================================================")

        if additional_info:
            log_str += f"\n{additional_info}"

        self.logger.info(log_str)
