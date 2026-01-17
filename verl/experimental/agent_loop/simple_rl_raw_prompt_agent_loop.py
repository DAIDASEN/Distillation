from __future__ import annotations

import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.dataset.simple_rl_dataset import extract_simple_rl_prompt_text
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("simple_rl_raw_prompt_agent")
class SimpleRLRawPromptAgentLoop(AgentLoopBase):
    """Single-turn agent loop that treats dataset `raw_prompt` as an already-formatted raw string."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        raw_prompt = kwargs["raw_prompt"]
        prompt_text = extract_simple_rl_prompt_text(raw_prompt)

        tokenized = self.tokenizer(prompt_text, add_special_tokens=False)
        prompt_ids = tokenized["input_ids"]

        if len(prompt_ids) > self.prompt_length:
            trunc = self.config.data.get("truncation", "error")
            if trunc == "error":
                raise ValueError(f"Prompt length {len(prompt_ids)} exceeds max_prompt_length={self.prompt_length}")
            if trunc == "left":
                prompt_ids = prompt_ids[-self.prompt_length :]
            else:  # "right" or unknown
                prompt_ids = prompt_ids[: self.prompt_length]

        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=None,
                video_data=None,
            )

        if output.num_preempted is not None:
            metrics["num_preempted"] = output.num_preempted

        response_mask = [1] * len(output.token_ids)
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=None,
            multi_modal_data={},
            num_turns=2,
            metrics=metrics,
        )

