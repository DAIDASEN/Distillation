from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


@dataclass
class _PendingTeacherBatch:
    batch: DataProto
    teacher_future: Any  # SimpleNamespace from recipe.gkd.teacher_utils.get_teacher_knowledge(is_async=True)


class SimpleRLDistillRayPPOTrainer(RayPPOTrainer):
    """RayPPOTrainer with optional teacher KD signals and one_step_off scheduling.

    This integrates a teacher service/client and attaches `teacher_topk_*` tensors to the
    batch so the actor can add a KD term to its loss.
    """

    def _distill_cfg(self):
        return self.config.actor_rollout_ref.actor.distill

    def _maybe_init_teacher(self):
        cfg = self._distill_cfg()
        if not cfg.enabled:
            self.teacher_client = None
            return
        from recipe.gkd.teacher import TeacherClient

        self.teacher_client = TeacherClient(
            server_ip=cfg.teacher_host,
            server_port=cfg.teacher_port,
            n_server_workers=cfg.n_server_workers,
            max_tokens=cfg.teacher_max_tokens,
            only_response=cfg.teacher_only_response,
        )

    def _submit_teacher(self, batch: DataProto, *, is_async: bool):
        from recipe.gkd.teacher_utils import get_teacher_knowledge

        cfg = self._distill_cfg()
        return get_teacher_knowledge(
            batch=batch,
            teacher_client=self.teacher_client,
            n_server_workers=cfg.n_server_workers,
            is_async=is_async,
        )

    def _attach_teacher_outputs(self, batch: DataProto, teacher_output: DataProto) -> DataProto:
        cfg = self._distill_cfg()
        if not cfg.enabled:
            return batch

        topk_logps_np = teacher_output.non_tensor_batch.get("teacher_topk_logps")
        topk_indices_np = teacher_output.non_tensor_batch.get("teacher_topk_indices")
        if topk_logps_np is None or topk_indices_np is None:
            return batch

        topk_logps = torch.from_numpy(topk_logps_np).to(torch.float32)
        topk_indices = torch.from_numpy(topk_indices_np).to(torch.int64)

        if topk_logps.dim() != 3 or topk_indices.dim() != 3:
            return batch

        if cfg.topk is not None and cfg.topk > 0 and topk_logps.size(-1) > cfg.topk:
            topk_logps = topk_logps[..., : cfg.topk].contiguous()
            topk_indices = topk_indices[..., : cfg.topk].contiguous()

        batch.batch["teacher_topk_logps"] = topk_logps
        batch.batch["teacher_topk_indices"] = topk_indices
        return batch

    def _train_one_batch(self, batch: DataProto, *, timing_raw: dict, metrics: dict, is_last_step: bool):
        # This is a minimally-copied subset of RayPPOTrainer.fit() inner body, operating on an already-generated batch.
        from verl.trainer.ppo.core_algos import compute_response_mask
        import ray
        from verl.trainer.ppo.reward import compute_reward_async
        from verl.trainer.ppo.ray_trainer import (
            AdvantageEstimator,
            apply_kl_penalty,
            compute_advantage,
            compute_data_metrics,
            compute_throughout_metrics,
            compute_timing_metrics,
            compute_variance_proxy_metrics,
            marked_timer,
            reduce_metrics,
        )

        # Ensure response_mask exists
        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)

        # Balance DP ranks if enabled.
        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics=metrics)

        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
        images_seqlens_all = []
        if "multi_modal_inputs" in batch.non_tensor_batch:
            for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                if not isinstance(multi_modal_input, dict) or "image_grid_thw" not in multi_modal_input:
                    continue
                images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
        batch.meta_info["images_seqlens"] = images_seqlens_all

        with marked_timer("reward", timing_raw, color="yellow"):
            if self.use_rm and "rm_scores" not in batch.batch.keys():
                if not self.use_reward_loop:
                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                else:
                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                    reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                batch = batch.union(reward_tensor)

            reward_tensor = None
            reward_extra_infos_dict: dict[str, list] = {}
            future_reward = None
            if self.config.reward_model.launch_reward_fn_async:
                future_reward = compute_reward_async.remote(data=batch, config=self.config, tokenizer=self.tokenizer)
            else:
                reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                    batch, reward_fn=self.reward_fn, reward_for_val=False
                )

        # Optional GRPO group filtering (sync-reward mode only, so we can filter before old_log_prob).
        if not self.config.reward_model.launch_reward_fn_async and reward_tensor is not None:
            batch.batch["token_level_scores"] = reward_tensor
            reward_extra_keys = list(reward_extra_infos_dict.keys()) if reward_extra_infos_dict else []
            if reward_extra_infos_dict:
                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

            if self.config.algorithm.adv_estimator == AdvantageEstimator.GRPO:
                from verl.trainer.ppo.group_filter import maybe_discard_grpo_trivial_groups

                batch, filter_metrics = maybe_discard_grpo_trivial_groups(
                    batch=batch,
                    token_level_scores=batch.batch["token_level_scores"],
                    rollout_n=self.config.actor_rollout_ref.rollout.n,
                    filter_groups_cfg=self.config.algorithm.filter_groups,
                )
                metrics.update(filter_metrics)
                reward_tensor = batch.batch["token_level_scores"]
                if reward_extra_keys:
                    reward_extra_infos_dict = {
                        k: batch.non_tensor_batch[k].tolist() for k in reward_extra_keys if k in batch.non_tensor_batch
                    }

            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        if bypass_recomputing_logprobs:
            from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

            apply_bypass_mode(
                batch=batch,
                rollout_corr_config=rollout_corr_config,
                policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
            )
        else:
            with marked_timer("old_log_prob", timing_raw, color="blue"):
                old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                entropys = old_log_prob.batch["entropys"]
                response_masks = batch.batch["response_mask"]
                actor_config = self.config.actor_rollout_ref.actor
                from verl.trainer.ppo.core_algos import agg_loss

                entropy_agg = agg_loss(
                    loss_mat=entropys,
                    loss_mask=response_masks,
                    loss_agg_mode=actor_config.loss_agg_mode,
                    loss_scale_factor=actor_config.loss_scale_factor,
                )
                metrics.update({"actor/entropy": entropy_agg.detach().item(), "perf/mfu/actor_infer": old_log_prob_mfu})
                old_log_prob.batch.pop("entropys")
                batch = batch.union(old_log_prob)

                if "rollout_log_probs" in batch.batch.keys():
                    from verl.utils.debug.metrics import calculate_debug_metrics

                    metrics.update(calculate_debug_metrics(batch))

        if self.use_reference_policy:
            with marked_timer("ref_log_prob", timing_raw, color="olive"):
                ref_log_prob = self._compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        if self.use_critic:
            with marked_timer("values", timing_raw, color="cyan"):
                values = self._compute_values(batch)
                batch = batch.union(values)

        with marked_timer("adv", timing_raw, color="brown"):
            if self.config.reward_model.launch_reward_fn_async:
                assert future_reward is not None
                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
            batch.batch["token_level_scores"] = reward_tensor
            if reward_extra_infos_dict:
                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

            if self.config.algorithm.use_kl_in_reward:
                batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                metrics.update(kl_metrics)
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=self.config.algorithm,
            )

        if self.use_critic:
            with marked_timer("update_critic", timing_raw, color="pink"):
                critic_output = self._update_critic(batch)
            metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

        if self.config.trainer.critic_warmup <= self.global_steps:
            with marked_timer("update_actor", timing_raw, color="red"):
                actor_output = self._update_actor(batch)
            metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

        # training metrics and logs
        metrics.update({"training/global_step": self.global_steps})
        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
        gradient_norm = metrics.get("actor/grad_norm", None)
        metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))

    def fit(self):
        cfg = self._distill_cfg()
        if not cfg.enabled:
            return super().fit()

        self._maybe_init_teacher()
        assert self.teacher_client is not None, "teacher_client init failed"

        # Reuse parent fit implementation but add teacher querying and optional one_step_off overlap.
        from omegaconf import OmegaConf
        from tqdm import tqdm
        from verl.utils.tracking import Tracking
        from verl.trainer.ppo.ray_trainer import marked_timer, RolloutSkip

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        current_epoch = self.global_steps // len(self.train_dataloader)

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            if val_metrics:
                logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # Match parent behavior: start from step 1.
        self.global_steps += 1

        pending: Optional[_PendingTeacherBatch] = None

        def post_train_step(metrics: dict[str, Any]):
            # Optional validation/checkpointing (same semantics as RayPPOTrainer.fit()).
            is_last_step = self.global_steps >= self.total_training_steps
            if (
                self.val_reward_fn is not None
                and self.config.trainer.test_freq > 0
                and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
            ):
                val_metrics = self._validate()
                if val_metrics:
                    metrics.update(val_metrics)

            if self.config.trainer.save_freq > 0 and (
                is_last_step or self.global_steps % self.config.trainer.save_freq == 0
            ):
                self._save_checkpoint()

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if self.global_steps > self.total_training_steps:
                    progress_bar.close()
                    return

                # 1) Build and rollout current batch
                metrics: dict[str, Any] = {}
                timing_raw: dict[str, Any] = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                batch.non_tensor_batch["uid"] = np.array(
                    [str(torch.randint(0, 2**31 - 1, (1,)).item()) for _ in range(len(batch.batch))],
                    dtype=object,
                )

                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, color="red"):
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                        gen_batch_output.meta_info.pop("timing", None)

                train_batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True).union(
                    gen_batch_output
                )

                teacher_future = self._submit_teacher(train_batch, is_async=cfg.async_enabled)
                current_pending = _PendingTeacherBatch(batch=train_batch, teacher_future=teacher_future)

                # 2) Train previous batch if available (one_step_off); otherwise sync on current.
                if cfg.async_enabled:
                    if pending is None:
                        pending = current_pending
                        continue

                    teacher_output_prev = pending.teacher_future.get()
                    prev_batch = self._attach_teacher_outputs(pending.batch, teacher_output_prev)

                    is_last_step = self.global_steps >= self.total_training_steps
                    with marked_timer("train", timing_raw, color="yellow"):
                        self._train_one_batch(prev_batch, timing_raw=timing_raw, metrics=metrics, is_last_step=is_last_step)

                    post_train_step(metrics)
                    logger.log(data=metrics, step=self.global_steps)
                    progress_bar.update(1)
                    self.global_steps += 1
                    pending = current_pending
                else:
                    teacher_output = teacher_future
                    train_batch = self._attach_teacher_outputs(train_batch, teacher_output)
                    is_last_step = self.global_steps >= self.total_training_steps
                    with marked_timer("train", timing_raw, color="yellow"):
                        self._train_one_batch(train_batch, timing_raw=timing_raw, metrics=metrics, is_last_step=is_last_step)
                    post_train_step(metrics)
                    logger.log(data=metrics, step=self.global_steps)
                    progress_bar.update(1)
                    self.global_steps += 1

            # flush pending at epoch end if async
            if cfg.async_enabled and pending is not None and self.global_steps <= self.total_training_steps:
                metrics = {}
                timing_raw = {}
                teacher_output_prev = pending.teacher_future.get()
                prev_batch = self._attach_teacher_outputs(pending.batch, teacher_output_prev)
                is_last_step = self.global_steps >= self.total_training_steps
                self._train_one_batch(prev_batch, timing_raw=timing_raw, metrics=metrics, is_last_step=is_last_step)
                post_train_step(metrics)
                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1
                pending = None
