"""OpenAI 兼容格式适配器（/v1/chat/completions）。"""
from __future__ import annotations

import time
from typing import Any

import httpx

from .base import Adapter, ChatMessage, ChatResponse, Usage


class OpenAIAdapter(Adapter):
    """对接 OpenAI 兼容格式的中转站。"""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120.0):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._client = httpx.Client(timeout=timeout)

    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        payload.update(kwargs)

        start = time.perf_counter()
        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]["message"]
        raw_usage = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=raw_usage.get("prompt_tokens"),
            completion_tokens=raw_usage.get("completion_tokens"),
            total_tokens=raw_usage.get("total_tokens"),
        )
        self._record_usage(usage)
        return ChatResponse(
            content=choice.get("content") or "",
            model=data.get("model"),
            usage=usage,
            raw=data,
            elapsed_ms=elapsed_ms,
        )
