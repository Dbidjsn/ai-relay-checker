"""编排：按顺序运行全部探针并汇总结果。"""
from __future__ import annotations

from .adapters import Adapter
from .probes.base import Probe, ProbeResult
from .probes.protocol import CharacterCountProbe, KnowledgeCutoffProbe, LatencyProbe
from .probes.fingerprint import (
    RandomNumberProbe,
    StylometryProbe,
    RefusalProbe,
    IdentityProbe,
)
from .probes.capability import (
    MathGradientProbe,
    CodeGradientProbe,
    ReasoningGradientProbe,
    ContextWindowProbe,
)


def all_probes(context_lengths: list[int] | None = None) -> list[Probe]:
    """返回当前已实现的所有探针。context_lengths 控制长上下文测试档位。"""
    return [
        CharacterCountProbe(),
        KnowledgeCutoffProbe(),
        LatencyProbe(),
        RandomNumberProbe(),
        StylometryProbe(),
        RefusalProbe(),
        IdentityProbe(),
        MathGradientProbe(),
        CodeGradientProbe(),
        ReasoningGradientProbe(),
        ContextWindowProbe(context_lengths),
    ]


def run_all(adapter: Adapter, context_lengths: list[int] | None = None) -> list[ProbeResult]:
    """运行全部探针，单个探针失败不中断整体。"""
    results: list[ProbeResult] = []
    for probe in all_probes(context_lengths):
        try:
            results.append(probe.run(adapter))
        except Exception as exc:  # noqa: BLE001 - 探针异常需兜底，继续跑后续
            results.append(
                ProbeResult(
                    probe=probe.name,
                    score=0.0,
                    verdict=f"失败: {exc}",
                    detail={"error": str(exc)},
                )
            )
    return results
