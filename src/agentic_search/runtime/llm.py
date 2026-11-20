"""LLM providers. The runtime only depends on :class:`LLMProvider`; swapping the backend (policy under
training, a strong teacher for distillation, a replay of a logged trajectory) is the "LLM Provider"
mechanism of the agent framework (PROPOSAL §3.1)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

import httpx

Messages = list[dict[str, str]]


class LLMProvider(Protocol):
    async def generate(
        self, messages: Messages, *, stop: Sequence[str] = (), assistant_prefix: str | None = None
    ) -> str:
        """Return the assistant continuation (excluding ``assistant_prefix``)."""
        ...


class OpenAICompatibleProvider:
    """Chat-completions client for vLLM / SGLang / hosted APIs.

    ``assistant_prefix`` uses vLLM's ``continue_final_message`` so a partially written assistant turn
    (e.g. an injected ``[Input Decomposition]``) is continued in-place rather than restarted.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        timeout_s: float = 120.0,
        extra_body: dict[str, Any] | None = None,
    ):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key = model, api_key
        self.params = {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
        self.timeout_s, self.extra_body = timeout_s, extra_body or {}

    async def generate(
        self, messages: Messages, *, stop: Sequence[str] = (), assistant_prefix: str | None = None
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            **self.params,
            **self.extra_body,
        }
        if stop:
            payload["stop"] = list(stop)
        if assistant_prefix:
            payload["messages"] = [*messages, {"role": "assistant", "content": assistant_prefix}]
            payload.update(continue_final_message=True, add_generation_prompt=False)
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(
                self.url, json=payload, headers={"Authorization": f"Bearer {self.api_key}"}
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"] or ""


class ScriptedProvider:
    """Deterministic provider for tests, demos and trajectory replay.

    ``script`` is either a list of turns (consumed in order) or ``fn(messages, assistant_prefix) -> str``.
    """

    def __init__(self, script: Sequence[str] | Callable[[Messages, str | None], str]):
        self._fn = script if callable(script) else None
        self._turns = list(script) if not callable(script) else []
        self.calls: list[tuple[Messages, str | None]] = []

    async def generate(
        self, messages: Messages, *, stop: Sequence[str] = (), assistant_prefix: str | None = None
    ) -> str:
        self.calls.append((list(messages), assistant_prefix))
        if self._fn is not None:
            text = self._fn(messages, assistant_prefix)
        elif self._turns:
            text = self._turns.pop(0)
        else:
            raise RuntimeError("ScriptedProvider ran out of turns")
        for s in stop:  # emulate server-side stop sequences
            if s in text:
                text = text[: text.index(s)]
        return text
