"""评分融合：多信号加权 → 掺水概率 → 结论分类。"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..probes.base import ProbeResult

# 各探针在掺水判定中的权重（总和 1.0）
WEIGHTS: dict[str, float] = {
    "字符计数": 0.10,
    "知识截止": 0.15,
    "延迟画像": 0.05,
    "随机数分布": 0.10,
    "写作风格": 0.05,
    "拒绝模式": 0.05,
    "身份一致性": 0.10,
    "数学梯度": 0.10,
    "代码梯度": 0.10,
    "推理梯度": 0.10,
    "长上下文": 0.10,
}

# 能力类信号 → 判「降智」
CAPABILITY_PROBES = {"字符计数", "数学梯度", "代码梯度", "推理梯度", "长上下文"}
# 身份类信号 → 判「替换」
IDENTITY_PROBES = {"身份一致性", "知识截止", "随机数分布"}


@dataclass
class Verdict:
    score: float            # 0~1 掺水概率
    level: str              # 结论分类
    substitution: float     # 替换维度分
    degradation: float      # 降智维度分
    notes: list[str] = field(default_factory=list)


def weigh(results: list[ProbeResult]) -> Verdict:
    """加权融合所有探针信号，输出 Verdict。"""
    total_weight = 0.0
    weighted = 0.0
    sub_w = sub_total = 0.0
    deg_w = deg_total = 0.0
    notes: list[str] = []

    for r in results:
        w = WEIGHTS.get(r.probe, 0.0)
        total_weight += w
        weighted += w * r.score
        if r.probe in CAPABILITY_PROBES:
            deg_w += w * r.score
            deg_total += w
        if r.probe in IDENTITY_PROBES:
            sub_w += w * r.score
            sub_total += w
        if r.score >= 0.5:
            notes.append(f"{r.probe}: {r.verdict}")

    score = weighted / total_weight if total_weight else 0.0
    degradation = deg_w / deg_total if deg_total else 0.0
    substitution = sub_w / sub_total if sub_total else 0.0

    if score >= 0.6:
        level = "疑似掺水"
    elif score >= 0.3:
        level = "可疑"
    else:
        level = "未发现明显掺水"

    # 仅在非正常时细分倾向：替换 vs 降智
    if score >= 0.3:
        level += "（偏向模型替换）" if substitution >= degradation else "（偏向降智）"

    return Verdict(
        score=score,
        level=level,
        substitution=substitution,
        degradation=degradation,
        notes=notes,
    )
