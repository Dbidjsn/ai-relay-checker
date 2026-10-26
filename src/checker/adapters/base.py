"""接入层统一抽象：定义消息、响应与适配器接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatMessage:
    """一条对话消息。role 取值：system / user / assistant。"""

    role: str
    content: str


@dataclass
class Usage:
    """token 用量（不同协议字段名不同，统一归一化）。"""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class ChatResponse:
    """一次对话请求的归一化结果。"""

    content: str
    model: str | None = None
    usage: Usage = field(default_factory=Usage)
    raw: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0


class Adapter(ABC):
    """中转站 API 适配器接口，OpenAI / Anthropic 协议各自实现。"""

    def __init__(self) -> None:
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    @property
    def total_tokens(self) -> int:
        """累计消耗的总 token 数。"""
        return self.total_prompt_tokens + self.total_completion_tokens

    def _record_usage(self, usage: Usage) -> None:
        self.total_prompt_tokens += usage.prompt_tokens or 0
        self.total_completion_tokens += usage.completion_tokens or 0

    @abstractmethod
    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        """发送对话请求并返回归一化响应。"""
        raise NotImplementedError
