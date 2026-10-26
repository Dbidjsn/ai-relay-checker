"""第 1 层：静态 / 协议探测（零基线，启动即用）。

识别「实际模型家族」与「是否量化」的硬信号，不依赖官方对照。
"""
from __future__ import annotations

import re

from ..adapters import Adapter, ChatMessage
from .base import Probe, ProbeResult


def extract_number(text: str) -> int | None:
    """从文本中提取第一个整数。"""
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else None


class CharacterCountProbe(Probe):
    """字符计数陷阱：弱模型 / 量化模型常数错单词里的字母数。"""

    name = "字符计数"
    description = "检验字符级感知能力，弱模型常出错"

    CASES = [
        ("单词 strawberry 里有几个字母 r？只回答一个数字。", 3),
        ("单词 hello 里有几个字母 l？只回答一个数字。", 2),
        ("单词 Mississippi 里有几个字母 s？只回答一个数字。", 4),
    ]

    def run(self, adapter: Adapter) -> ProbeResult:
        results = []
        for question, expected in self.CASES:
            text = self.ask(adapter, question)
            got = extract_number(text)
            results.append(
                {"question": question, "expected": expected, "got": got, "ok": got == expected}
            )
        ok = sum(1 for r in results if r["ok"])
        score = round(1 - ok / len(results), 2)
        return ProbeResult(
            probe=self.name,
            score=score,
            verdict=f"答对 {ok}/{len(results)} 题",
            detail={"cases": results},
        )


class KnowledgeCutoffProbe(Probe):
    """知识截止点：模型能否准确说出近期事件年份，反映训练数据新旧。

    旧模型（或换壳的国产老模型）冒充新模型时，会答不出较新的事件。
    """

    name = "知识截止"
    description = "探测训练数据时间窗，旧模型冒充新模型时会露馅"

    MILESTONES = [
        ("GPT-4 是哪一年发布的？只回答年份。", 2023),
        ("GPT-4o 是哪一年发布的？只回答年份。", 2024),
        ("Anthropic 的 Claude 3 系列是哪一年发布的？只回答年份。", 2024),
    ]

    def run(self, adapter: Adapter) -> ProbeResult:
        results = []
        for question, expected in self.MILESTONES:
            text = self.ask(adapter, question)
            got = extract_number(text)
            results.append(
                {"question": question, "expected": expected, "got": got, "ok": got == expected}
            )
        ok = sum(1 for r in results if r["ok"])
        score = round(1 - ok / len(results), 2)
        return ProbeResult(
            probe=self.name,
            score=score,
            verdict=f"命中 {ok}/{len(results)} 个时间点",
            detail={"milestones": results},
        )


class LatencyProbe(Probe):
    """延迟 / 速度画像：量化小模型输出速度常异常偏高。

    本探针只记录数据，速度是否异常由评分层结合其他信号判定。
    """

    name = "延迟画像"
    description = "记录响应耗时与输出速率，作为量化/低端硬件的弱信号"

    def run(self, adapter: Adapter) -> ProbeResult:
        resp = adapter.chat([ChatMessage("user", "请用 200 字介绍一下北京。")])
        elapsed_ms = resp.elapsed_ms
        out_tokens = resp.usage.completion_tokens or 0
        tps = out_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0.0
        return ProbeResult(
            probe=self.name,
            score=0.0,
            verdict=f"耗时 {elapsed_ms:.0f}ms，输出 {out_tokens} token（{tps:.1f} tok/s）",
            detail={
                "elapsed_ms": elapsed_ms,
                "completion_tokens": out_tokens,
                "tps": tps,
            },
        )
