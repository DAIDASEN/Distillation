from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from verl import DataProto


def maybe_discard_grpo_trivial_groups(
    *,
    batch: DataProto,
    token_level_scores: torch.Tensor,
    rollout_n: int,
    filter_groups_cfg: Optional[Any],
    atol: float = 1e-6,
) -> tuple[DataProto, dict[str, Any]]:
    """Drop trivial GRPO groups (all-correct or all-wrong) from the batch.

    Motivation: for outcome-only GRPO with binary rewards, groups where all samples are correct
    or all samples are wrong provide no preference signal and can be skipped to avoid updates
    dominated by regularizers (entropy/KL/KD).

    This function returns a *new* DataProto with discarded samples removed (via `DataProto.select_idxs`).
    If everything would be discarded, it keeps one full group to avoid creating an empty batch.
    """
    if filter_groups_cfg is None or not getattr(filter_groups_cfg, "enable", False):
        return batch, {}

    metric = getattr(filter_groups_cfg, "metric", None)
    if metric not in (None, "acc"):
        return batch, {}

    if "uid" not in batch.non_tensor_batch:
        return batch, {}
    if "response_mask" not in batch.batch:
        return batch, {}

    uids = batch.non_tensor_batch["uid"]
    if uids is None:
        return batch, {}

    seq_scores = token_level_scores.detach().float().sum(dim=-1).cpu().numpy()
    bsz = int(seq_scores.shape[0])

    uid2idxs: dict[Any, list[int]] = {}
    for i in range(bsz):
        uid2idxs.setdefault(uids[i], []).append(i)

    discard_mask = np.zeros((bsz,), dtype=bool)
    discarded_groups = 0
    for _, idxs in uid2idxs.items():
        if rollout_n > 0 and len(idxs) != rollout_n:
            continue
        vals = seq_scores[idxs]
        if np.allclose(vals, 0.0, atol=atol) or np.allclose(vals, 1.0, atol=atol):
            discard_mask[idxs] = True
            discarded_groups += 1

    if discard_mask.all():
        # Keep one full group to avoid empty batch (which can break loss aggregation).
        for _, idxs in uid2idxs.items():
            if rollout_n > 0 and len(idxs) != rollout_n:
                continue
            discard_mask[idxs] = False
            discarded_groups = max(0, discarded_groups - 1)
            break

    if discard_mask.any():
        keep_mask = ~discard_mask
        batch = batch.select_idxs(keep_mask)

    kept = int((~discard_mask).sum())
    discarded = int(discard_mask.sum())
    return batch, {
        "filter_groups/metric": metric or "acc",
        "filter_groups/discarded_groups": int(discarded_groups),
        "filter_groups/discarded_samples": discarded,
        "filter_groups/kept_samples": kept,
        "filter_groups/discarded_frac": float(discarded / max(1, discarded + kept)),
    }
