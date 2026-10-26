"""统计量工具：KL / JS 散度，用于指纹库匹配（预留）。"""
from __future__ import annotations

import math


def kl_divergence(p: list[float], q: list[float]) -> float:
    """KL 散度 D(p||q)，对 q 中的零概率做平滑。"""
    total = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0:
            continue
        if qi <= 0:
            qi = 1e-12
        total += pi * math.log(pi / qi)
    return total


def jensen_shannon(p: list[float], q: list[float]) -> float:
    """Jensen-Shannon 散度，对称，取值 [0, ln2]。"""
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
