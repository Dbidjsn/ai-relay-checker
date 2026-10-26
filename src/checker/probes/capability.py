"""第 3 层：能力梯度探测（判断「降智 / 量化」）。

量化降智的典型特征：简单题会、难题崩、长文失忆。
崩点越靠前，越可疑。
"""
from __future__ import annotations

from ..adapters import Adapter
from .base import Probe, ProbeResult
from .protocol import extract_number


class GradientProbe(Probe):
    """能力梯度探针基类：题集从易到难，崩点越靠前越可疑。

    子类覆盖 cases：[(难度标签, 题目, 预期数字答案), ...]
    """

    cases: list[tuple[str, str, int]] = []

    def run(self, adapter: Adapter) -> ProbeResult:
        results = []
        for level, question, expected in self.cases:
            text = self.ask(adapter, question)
            got = extract_number(text)
            results.append(
                {"level": level, "expected": expected, "got": got, "ok": got == expected}
            )

        total = len(results)
        ok_count = sum(1 for r in results if r["ok"])
        first_wrong = next(
            (i for i, r in enumerate(results) if not r["ok"]), total
        )

        if first_wrong == total:
            score, note = 0.0, "全部通过"
        elif first_wrong == 0:
            score, note = 0.9, "基础题即答错（疑似严重降智）"
        elif first_wrong == 1:
            score, note = 0.6, "简单题答错"
        elif first_wrong == 2:
            score, note = 0.3, "中档题答错"
        else:
            score, note = 0.15, "仅高档题答错"

        return ProbeResult(
            probe=self.name,
            score=score,
            verdict=f"{note}：通过 {ok_count}/{total} 档",
            detail={"cases": results, "first_wrong_level": first_wrong},
        )


class MathGradientProbe(GradientProbe):
    name = "数学梯度"
    description = "从算术到高阶求和，检验数学能力是否被降智"
    cases = [
        ("基础", "7 加 8 等于多少？只回答数字。", 15),
        ("初级", "一个数的 30% 是 60，这个数是多少？只回答数字。", 200),
        ("中级", "方程 2x + 5 = 13 中 x 等于多少？只回答数字。", 4),
        ("高级", "1 到 100 所有整数的和是多少？只回答数字。", 5050),
    ]


class CodeGradientProbe(GradientProbe):
    name = "代码梯度"
    description = "检验 Python 基础与计算能力"
    cases = [
        ("基础", "Python 中 len('hello') 的值是多少？只回答数字。", 5),
        ("初级", "Python 中 2 ** 10 的值是多少？只回答数字。", 1024),
        ("中级", "Python 中 sum(range(101)) 的值是多少？只回答数字。", 5050),
        ("高级", "Python 中 3 * (2 + 4) 的值是多少？只回答数字。", 18),
    ]


class ReasoningGradientProbe(GradientProbe):
    name = "推理梯度"
    description = "从单步到多步文字推理，检验逻辑能力"
    cases = [
        ("基础", "小明有 5 个苹果，吃了 2 个，还剩几个？只回答数字。", 3),
        ("初级", "一个班有 20 人，男生比女生多 4 人，男生有几人？只回答数字。", 12),
        ("中级", "甲、乙相距 100 米相向而行，甲每秒 3 米、乙每秒 2 米，几秒后相遇？只回答数字。", 20),
        ("高级", "三个连续整数之和是 36，最大的那个是多少？只回答数字。", 13),
    ]


class ContextWindowProbe(Probe):
    """长上下文窗口：量化/缩水模型会在较短长度就「失忆」。

    默认测 8000 字符（约 8K token）一档；可传入多档（如 [8000, 32000]），
    从短到长渐进测试，短档失忆即提前停止以节省 token。
    """

    name = "长上下文"
    description = "在长文开头埋密码，末尾提问，检验有效上下文长度"

    def __init__(self, lengths: list[int] | None = None):
        self.lengths = lengths or [8000]

    def run(self, adapter: Adapter) -> ProbeResult:
        secret = "ZQ7-M42"
        levels = []
        for chars in self.lengths:
            prompt = (
                f"请记住这个密码：{secret}。之后我会问你。\n\n"
                f"{self._make_filler(chars)}\n"
                f"现在告诉我：我开头让你记住的密码是什么？只回答密码本身。"
            )
            text = self.ask(adapter, prompt)
            ok = secret.lower() in text.lower()
            levels.append({"chars": chars, "recalled": ok, "answer": text[:40]})
            if not ok:
                break  # 短档失忆，更长档大概率也失忆，提前停止省 token

        first_ok = levels[0]["recalled"] if levels else True
        all_ok = all(lv["recalled"] for lv in levels)
        if not first_ok:
            score = 0.7
        elif not all_ok:
            score = 0.3
        else:
            score = 0.0

        passed = sum(1 for lv in levels if lv["recalled"])
        verdict = f"通过 {passed}/{len(levels)} 档（最长 {levels[-1]['chars']} 字符）"
        return ProbeResult(
            probe=self.name,
            score=score,
            verdict=verdict,
            detail={"secret": secret, "levels": levels},
        )

    @staticmethod
    def _make_filler(target_chars: int) -> str:
        unit = "这是用来填充上下文的无关文本。\n"
        return unit * (target_chars // len(unit) + 1)
