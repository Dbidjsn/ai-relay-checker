"""探针框架：每个探针输出一个 0~1 的可疑度信号。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..adapters import Adapter, ChatMessage


@dataclass
class ProbeResult:
    """单个探针的检测结果。

    score 语义统一为「掺水可疑度」：0 = 正常，1 = 高度可疑。
    """

    probe: str
    score: float
    verdict: str
    detail: dict[str, Any] = field(default_factory=dict)


class Probe(ABC):
    """探针基类。子类实现 run，返回一个 ProbeResult。"""

    name: str = ""
    description: str = ""

    @abstractmethod
    def run(self, adapter: Adapter) -> ProbeResult:
        raise NotImplementedError

    def ask(self, adapter: Adapter, prompt: str) -> str:
        """发送一条 user 消息，返回纯文本回复。"""
        resp = adapter.chat([ChatMessage("user", prompt)])
        return resp.content
