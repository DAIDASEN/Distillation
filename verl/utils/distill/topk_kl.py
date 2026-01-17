from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SparseKLLossOutput:
    kl_per_token: torch.Tensor  # [B, T]


def _teacher_topk_probs_from_logps(
    logps: torch.Tensor, *, normalize: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert top-k log-probabilities to probabilities.

    NOTE: teacher logps are typically log-softmax values from the *full* vocab distribution, truncated
    to top-k. In that case, the exp(logps) mass will be < 1.0. For "sparse KL" we usually do NOT
    renormalize (it approximates the full KL by summing over top-k support only).
    """
    probs = torch.exp(logps)
    if normalize:
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-20)
        logps = torch.log(probs + 1e-20)
    return probs, logps


def sparse_kl_from_logits_topk(
    *,
    logits: torch.Tensor,  # [B, T, V]
    teacher_topk_indices: torch.Tensor,  # [B, T, K]
    teacher_topk_logps: torch.Tensor,  # [B, T, K]
    mask: Optional[torch.Tensor] = None,  # [B, T] bool
    chunk_size: int = 2048,
    normalize_teacher: bool = False,
) -> SparseKLLossOutput:
    """Sparse KL(teacher || student) on teacher top-k support.

    Computes: KL = Σ_{i in topk} p_t(i) * (log p_t(i) - log p_s(i))
    where p_s(i) is the student's full softmax probability at token i.

    This avoids materializing full log-softmax by using:
      log p_s(i) = logits_i - logsumexp(logits)
    """
    if logits.dim() != 3:
        raise ValueError(f"Expected logits [B,T,V], got {tuple(logits.shape)}")
    if teacher_topk_indices.shape[:2] != logits.shape[:2] or teacher_topk_logps.shape[:2] != logits.shape[:2]:
        raise ValueError(
            "teacher_topk_* must match logits [B,T,*], got "
            f"{tuple(teacher_topk_indices.shape)=} {tuple(teacher_topk_logps.shape)=} {tuple(logits.shape)=}"
        )
    if teacher_topk_indices.shape[-1] != teacher_topk_logps.shape[-1]:
        raise ValueError("teacher_topk_indices and teacher_topk_logps must have same last dim")

    device = logits.device
    B, T, V = logits.shape
    K = teacher_topk_indices.shape[-1]

    flat_logits = logits.reshape(B * T, V)
    flat_idx = teacher_topk_indices.reshape(B * T, K).to(torch.long)
    flat_logps_t = teacher_topk_logps.reshape(B * T, K)

    if mask is None:
        active = torch.ones((B * T,), dtype=torch.bool, device=device)
    else:
        active = mask.reshape(B * T).to(device=device, dtype=torch.bool)

    kl_flat = torch.zeros((B * T,), dtype=flat_logits.dtype, device=device)
    active_idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
    if active_idx.numel() == 0:
        return SparseKLLossOutput(kl_per_token=kl_flat.reshape(B, T))

    for s in range(0, active_idx.numel(), chunk_size):
        sel = active_idx[s : s + chunk_size]
        chunk_logits = flat_logits[sel]
        chunk_idx = flat_idx[sel]
        chunk_logps_t = flat_logps_t[sel]

        probs_t, logps_t = _teacher_topk_probs_from_logps(
            chunk_logps_t.to(torch.float32), normalize=normalize_teacher
        )

        logits_topk = torch.gather(chunk_logits, dim=-1, index=chunk_idx)
        logZ = torch.logsumexp(chunk_logits.to(torch.float32), dim=-1, keepdim=True)
        logps_s_topk = logits_topk.to(torch.float32) - logZ

        kl = torch.sum(probs_t * (logps_t - logps_s_topk), dim=-1)
        kl_flat[sel] = kl.to(kl_flat.dtype)

    return SparseKLLossOutput(kl_per_token=kl_flat.reshape(B, T))
