from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch

from verl.utils.dataset.rl_dataset import RLHFDataset


def extract_simple_rl_prompt_text(prompt: Any) -> str:
    """Extract raw ChatML-like prompt string from SimpleRL-Zoo parquet column.

    Verified format:
      - prompt is numpy.ndarray with shape (1,)
      - prompt[0] is a dict with keys {"role","content"}
      - prompt[0]["content"] is a fully formatted raw string ending with "<|im_start|>assistant\\n"
    """
    if prompt is None:
        raise ValueError("prompt is None")

    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()

    if isinstance(prompt, str):
        return prompt

    # Most common: list with a single dict element.
    if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], dict) and "content" in prompt[0]:
        content = prompt[0]["content"]
        if not isinstance(content, str):
            raise TypeError(f"prompt[0]['content'] must be str, got {type(content)}")
        return content

    # Sometimes parquet decoding may produce a dict directly.
    if isinstance(prompt, dict) and "content" in prompt:
        content = prompt["content"]
        if not isinstance(content, str):
            raise TypeError(f"prompt['content'] must be str, got {type(content)}")
        return content

    raise TypeError(f"Unsupported SimpleRL prompt type/shape: {type(prompt)}")


def _normalize_reward_model(reward_model: Any, gt_answer: Any) -> dict[str, Any]:
    if reward_model is None or reward_model == "":
        reward_model = {}

    if isinstance(reward_model, str):
        try:
            reward_model = json.loads(reward_model)
        except Exception:
            reward_model = {}

    if not isinstance(reward_model, dict):
        reward_model = {}

    if "ground_truth" not in reward_model or reward_model["ground_truth"] is None:
        if gt_answer is not None:
            reward_model["ground_truth"] = str(gt_answer)
    else:
        reward_model["ground_truth"] = str(reward_model["ground_truth"])

    return reward_model


class SimpleRLRLDataset(RLHFDataset):
    """RLHFDataset adapter for SimpleRL-Zoo parquet.

    Key difference vs RLHFDataset:
      - returns `raw_prompt` as a raw string (already ChatML-formatted)
      - does NOT apply any chat template in dataset/rollout
    """

    def maybe_filter_out_long_prompts(self, dataframe=None):
        if dataframe is None:
            dataframe = self.dataframe

        if not self.filter_overlong_prompts:
            return dataframe

        tokenizer = self.tokenizer
        max_prompt_length = self.max_prompt_length
        prompt_key = self.prompt_key

        def doc2len(doc) -> int:
            try:
                prompt_text = extract_simple_rl_prompt_text(doc[prompt_key])
                return len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
            except Exception:
                return max_prompt_length + 1

        return dataframe.filter(
            lambda doc: doc2len(doc) <= max_prompt_length,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {max_prompt_length} tokens (SimpleRL raw prompt)",
        )

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]

        prompt_text = extract_simple_rl_prompt_text(row_dict.get(self.prompt_key))
        row_dict["raw_prompt"] = prompt_text

        row_dict["reward_model"] = _normalize_reward_model(row_dict.get("reward_model"), row_dict.get("gt_answer"))
        if "data_source" not in row_dict or row_dict["data_source"] is None:
            row_dict["data_source"] = "simpleRL"

        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = {}
        extra_info = row_dict["extra_info"]
        if not isinstance(extra_info, dict):
            extra_info = {}
            row_dict["extra_info"] = extra_info

        index = extra_info.get("index", 0)
        row_dict["index"] = index
        row_dict["tools_kwargs"] = extra_info.get("tools_kwargs", {})
        row_dict["interaction_kwargs"] = extra_info.get("interaction_kwargs", {})

        return row_dict
