"""Anthropic 兼容格式适配器（/v1/messages）。"""
from __future__ import annotations

import time
from typing import Any

import httpx

from .base import Adapter, ChatMessage, ChatResponse, Usage


class AnthropicAdapter(Adapter):
    """对接 Anthropic 兼容格式的中转站。"""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120.0):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._client = httpx.Client(timeout=timeout)

    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        # Anthropic 协议把 system 单独放到顶层字段
        system_parts = [m.content for m in messages if m.role == "system"]
        rest = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role != "system"
        ]

        payload: dict[str, Any] = {"model": self.model, "messages": rest}
        if system_parts:
            payload["system"] = "\n".join(system_parts)
        payload["max_tokens"] = kwargs.pop("max_tokens", 1024)
        payload.update(kwargs)

        start = time.perf_counter()
        resp = self._client.post(
            f"{self.base_url}/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            json=payload,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        data = resp.json()

        content = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )
        raw_usage = data.get("usage") or {}
        prompt_tokens = raw_usage.get("input_tokens")
        completion_tokens = raw_usage.get("output_tokens")
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=(
                (prompt_tokens or 0) + (completion_tokens or 0)
                if prompt_tokens is not None and completion_tokens is not None
                else None
            ),
        )
        self._record_usage(usage)
        return ChatResponse(
            content=content,
            model=data.get("model"),
            usage=usage,
            raw=data,
            elapsed_ms=elapsed_ms,
        )
