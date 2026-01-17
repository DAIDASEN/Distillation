from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from reward.simpleRL_rule_math import compute_reward
from verl import DataProto
from verl.experimental.dataset.simple_rl_dataset import extract_simple_rl_prompt_text
from verl.trainer.ppo.group_filter import maybe_discard_grpo_trivial_groups
from verl.utils.distill.topk_kl import sparse_kl_from_logits_topk


def test_reward_simple_rl_math_boxed():
    assert compute_reward(solution_str=r"final answer: \boxed{118}", ground_truth="118") == 1.0
    assert compute_reward(solution_str=r"final answer: \boxed{119}", ground_truth="118") == 0.0
    assert compute_reward(solution_str="no box", ground_truth="118") == 0.0


def test_extract_simple_rl_prompt_text_from_parquet_row():
    parquet = Path("data/simpleRL/simplelr_qwen_level3to5/train.parquet")
    if not parquet.exists():
        pytest.skip(f"missing parquet: {parquet}")

    import pandas as pd

    df = pd.read_parquet(parquet, columns=["prompt"])
    prompt = df.iloc[0]["prompt"]

    text = extract_simple_rl_prompt_text(prompt)

    if isinstance(prompt, np.ndarray):
        expected = prompt.tolist()[0]["content"]
    else:
        expected = prompt[0]["content"]
    assert text == expected
    assert isinstance(text, str)
    assert text.endswith("<|im_start|>assistant\n")


def test_sparse_kd_kl_loss_cpu_backward():
    torch.manual_seed(0)
    B, T, V, K = 2, 3, 11, 4
    logits = torch.randn(B, T, V, dtype=torch.float32, requires_grad=True)
    teacher_topk_indices = torch.randint(0, V, (B, T, K), dtype=torch.int64)
    teacher_topk_logps = torch.randn(B, T, K, dtype=torch.float32)
    teacher_topk_logps = torch.log_softmax(teacher_topk_logps, dim=-1)
    response_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)

    out = sparse_kl_from_logits_topk(
        logits=logits,
        teacher_topk_indices=teacher_topk_indices,
        teacher_topk_logps=teacher_topk_logps,
        mask=response_mask,
        chunk_size=8,
    )
    loss = out.kl_per_token[response_mask].mean()
    assert torch.isfinite(loss).item()
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all().item()


def test_grpo_trivial_group_filter_drops_all_correct_or_all_wrong_groups():
    from types import SimpleNamespace

    rollout_n = 4
    T = 3

    # 2 groups * 4 samples each
    uids = np.array(["q1"] * rollout_n + ["q2"] * rollout_n, dtype=object)
    response_mask = torch.ones((2 * rollout_n, T), dtype=torch.bool)
    token_level_scores = torch.zeros((2 * rollout_n, T), dtype=torch.float32)
    # q1: all wrong (sum=0)
    # q2: mixed (some 0, some 1)
    token_level_scores[rollout_n + 0, -1] = 0.0
    token_level_scores[rollout_n + 1, -1] = 1.0
    token_level_scores[rollout_n + 2, -1] = 0.0
    token_level_scores[rollout_n + 3, -1] = 1.0

    batch = DataProto.from_single_dict({"response_mask": response_mask, "uid": uids})
    cfg = SimpleNamespace(enable=True, metric="acc")

    filtered, metrics = maybe_discard_grpo_trivial_groups(
        batch=batch, token_level_scores=token_level_scores, rollout_n=rollout_n, filter_groups_cfg=cfg
    )
    assert len(filtered) == rollout_n
    assert set(filtered.non_tensor_batch["uid"].tolist()) == {"q2"}
    assert metrics["filter_groups/discarded_groups"] == 1
    assert metrics["filter_groups/discarded_samples"] == rollout_n


def test_grpo_trivial_group_filter_never_returns_empty_batch():
    from types import SimpleNamespace

    rollout_n = 4
    T = 2
    uids = np.array(["q1"] * rollout_n + ["q2"] * rollout_n, dtype=object)
    response_mask = torch.ones((2 * rollout_n, T), dtype=torch.bool)
    token_level_scores = torch.zeros((2 * rollout_n, T), dtype=torch.float32)
    # Both groups are all-wrong => would discard all, but should keep one group for stability.
    batch = DataProto.from_single_dict({"response_mask": response_mask, "uid": uids})
    cfg = SimpleNamespace(enable=True, metric="acc")

    filtered, metrics = maybe_discard_grpo_trivial_groups(
        batch=batch, token_level_scores=token_level_scores, rollout_n=rollout_n, filter_groups_cfg=cfg
    )
    assert len(filtered) == rollout_n
    assert metrics["filter_groups/kept_samples"] == rollout_n
