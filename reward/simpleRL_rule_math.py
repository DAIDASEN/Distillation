from __future__ import annotations

import re
from typing import Any, Optional


_BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")


def _extract_boxed_answer(text: str) -> Optional[str]:
    if not text:
        return None
    matches = _BOX_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def compute_reward(solution_str: str, ground_truth: str, **_: Any) -> float:
    """SimpleRL math rule reward: 1.0 if the final \\boxed{...} matches ground truth else 0.0."""
    gt = "" if ground_truth is None else str(ground_truth).strip()
    pred = _extract_boxed_answer(solution_str or "")
    return 1.0 if pred is not None and pred == gt else 0.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs: Any,
) -> float:
    # Keep signature compatible with verl.utils.reward_score.default_compute_score.
    return compute_reward(solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info, **kwargs)
