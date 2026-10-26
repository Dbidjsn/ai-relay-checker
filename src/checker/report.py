"""报告生成：终端 / JSON / HTML。"""
from __future__ import annotations

from .probes.base import ProbeResult
from .scoring.verdict import Verdict


def format_terminal(
    results: list[ProbeResult],
    verdict: Verdict,
    target_model: str,
    base_url: str,
    total_tokens: int = 0,
) -> str:
    lines = [
        f"检测目标: {target_model} @ {base_url}",
        f"总消耗:   {total_tokens} token",
        "=" * 50,
    ]
    for r in results:
        flag = "⚠" if r.score >= 0.5 else "✓"
        lines.append(f"[{flag}] {r.probe}: {r.verdict}  (可疑度 {r.score:.2f})")
    lines.append("-" * 50)
    lines.append(f"掺水概率: {verdict.score:.0%}")
    lines.append(f"结论:     {verdict.level}")
    lines.append(f"维度:     替换 {verdict.substitution:.0%} · 降智 {verdict.degradation:.0%}")
    if verdict.notes:
        lines.append("可疑项:")
        lines.extend(f"  - {n}" for n in verdict.notes)
    return "\n".join(lines)


def to_json(
    results: list[ProbeResult],
    verdict: Verdict,
    target_model: str,
    base_url: str,
    total_tokens: int = 0,
) -> dict:
    return {
        "target": {"model": target_model, "base_url": base_url},
        "usage": {"total_tokens": total_tokens},
        "verdict": {
            "score": round(verdict.score, 3),
            "level": verdict.level,
            "substitution": round(verdict.substitution, 3),
            "degradation": round(verdict.degradation, 3),
        },
        "probes": [
            {
                "probe": r.probe,
                "score": r.score,
                "verdict": r.verdict,
                "detail": r.detail,
            }
            for r in results
        ],
    }


def to_html(
    results: list[ProbeResult],
    verdict: Verdict,
    target_model: str,
    base_url: str,
    total_tokens: int = 0,
) -> str:
    rows = "".join(
        f"<tr><td>{r.probe}</td><td>{r.score:.2f}</td><td>{r.verdict}</td></tr>"
        for r in results
    )
    if verdict.score >= 0.6:
        color = "#d32f2f"
    elif verdict.score >= 0.3:
        color = "#f57c00"
    else:
        color = "#2e7d32"
    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>掺水检测报告</title>
<style>
body{{font-family:sans-serif;max-width:820px;margin:40px auto;padding:0 20px;color:#222}}
h1{{font-size:1.5em}} .score{{font-size:2.6em;font-weight:bold;color:{color}}}
table{{border-collapse:collapse;width:100%;margin-top:20px}}
td,th{{border:1px solid #ddd;padding:8px 10px;text-align:left}}
th{{background:#f5f5f5}}
</style></head><body>
<h1>AI 中转站掺水检测报告</h1>
<p>目标：{target_model} @ {base_url} · 总消耗 {total_tokens} token</p>
<p>掺水概率：<span class="score">{verdict.score:.0%}</span></p>
<p>结论：{verdict.level}</p>
<p>替换维度 {verdict.substitution:.0%} · 降智维度 {verdict.degradation:.0%}</p>
<table><tr><th>探针</th><th>可疑度</th><th>结论</th></tr>{rows}</table>
</body></html>"""
