"""Teacher-forcing log-probs for OPD from a separately deployed vLLM server.

The frozen teacher (Qwen3-235B-A22B) runs on its own vLLM cluster. After the student finishes a rollout,
the full token sequence is sent once; the teacher scores it in a single prefill pass
(``prompt_logprobs``) and we keep the log-prob of each *sampled* token -- no vocabulary-sized
distributions cross the network.

Requirement: teacher and student must share the tokenizer (true within the Qwen3 family). With a
different vocabulary, sampled-token reverse KL is undefined and a cross-tokenizer alignment is needed.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import torch


def parse_prompt_logprobs(entries: Sequence[Any], token_ids: Sequence[int]) -> list[float]:
    """vLLM returns, per prompt position, ``None`` (position 0) or ``{token_id: {"logprob": ...}}``."""
    out: list[float] = []
    for pos, (entry, tok) in enumerate(zip(entries, token_ids)):
        if entry is None:
            out.append(0.0 if pos == 0 else float("nan"))
            continue
        info = entry.get(str(tok), entry.get(tok)) if isinstance(entry, dict) else None
        out.append(float(info["logprob"]) if isinstance(info, dict) else float("nan"))
    return out


class TeacherLogprobClient:
    def __init__(self, base_url: str, model: str, max_workers: int = 32, timeout_s: float = 300.0):
        self.url = base_url.rstrip("/") + "/completions"
        self.model, self.max_workers, self.timeout_s = model, max_workers, timeout_s

    def _score_one(self, prompt_ids: list[int], response_ids: list[int]) -> list[float]:
        ids = list(prompt_ids) + list(response_ids)
        payload = {
            "model": self.model,
            "prompt": ids,
            "max_tokens": 1,
            "temperature": 0.0,
            "prompt_logprobs": 0,
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.post(self.url, json=payload)
            r.raise_for_status()
            entries = r.json()["choices"][0]["prompt_logprobs"]
        return parse_prompt_logprobs(entries, ids)[len(prompt_ids) :]

    def score(self, prompts: Sequence[list[int]], responses: Sequence[list[int]]) -> list[list[float]]:
        with ThreadPoolExecutor(self.max_workers) as pool:
            return list(pool.map(self._score_one, prompts, responses))

    def score_padded(
        self, prompts: Sequence[list[int]], responses: Sequence[list[int]], length: int
    ) -> torch.Tensor:
        """``[B, length]`` right-padded tensor aligned with VERL's response tensors (NaN -> 0)."""
        out = torch.zeros(len(responses), length)
        for i, lp in enumerate(self.score(prompts, responses)):
            t = torch.tensor(lp[:length])
            out[i, : len(t)] = torch.nan_to_num(t, nan=0.0)
        return out
