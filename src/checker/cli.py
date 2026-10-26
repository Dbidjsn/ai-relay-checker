"""CLI 入口：向中转站发送探测请求并打印结果。"""
from __future__ import annotations

import argparse
import json
import sys

from .adapters import ChatMessage, OpenAIAdapter, AnthropicAdapter
from .runner import run_all
from .scoring.verdict import weigh
from .report import format_terminal, to_json, to_html


def add_common(parser: argparse.ArgumentParser) -> None:
    """给子命令添加公共的接入参数。"""
    parser.add_argument(
        "--provider", choices=["openai", "anthropic"], default="openai",
        help="协议类型（默认 openai）",
    )
    parser.add_argument(
        "--base-url", required=True,
        help="中转站地址（OpenAI 到 /v1，Anthropic 到根路径）",
    )
    parser.add_argument("--api-key", required=True, help="API key")
    parser.add_argument("--model", required=True, help="声称的模型名")


def build_adapter(args: argparse.Namespace):
    if args.provider == "openai":
        return OpenAIAdapter(args.base_url, args.api_key, args.model)
    if args.provider == "anthropic":
        return AnthropicAdapter(args.base_url, args.api_key, args.model)
    sys.exit(f"不支持的 provider: {args.provider}")


def parse_lengths(text: str | None) -> list[int] | None:
    """把逗号分隔的档位字符串解析成 int 列表。"""
    if not text:
        return None
    return [int(x) for x in text.split(",") if x.strip()]


def cmd_chat(args: argparse.Namespace) -> None:
    adapter = build_adapter(args)
    resp = adapter.chat([ChatMessage("user", args.prompt)])
    print(f"返回 model: {resp.model}")
    print(f"耗时:      {resp.elapsed_ms:.0f} ms")
    print(f"usage:     {resp.usage}")
    print("回复:")
    print(resp.content)


def cmd_scan(args: argparse.Namespace) -> None:
    adapter = build_adapter(args)
    lengths = parse_lengths(args.context_lengths)
    results = run_all(adapter, context_lengths=lengths)
    verdict = weigh(results)
    tokens = adapter.total_tokens
    print(format_terminal(results, verdict, args.model, args.base_url, tokens))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(
                to_json(results, verdict, args.model, args.base_url, tokens),
                f, ensure_ascii=False, indent=2,
            )
        print(f"\n已导出 JSON: {args.json}")
    if args.html:
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(to_html(results, verdict, args.model, args.base_url, tokens))
        print(f"已导出 HTML: {args.html}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="checker", description="AI 中转站掺水检测器"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    chat_p = sub.add_parser("chat", help="发送单条测试消息")
    add_common(chat_p)
    chat_p.add_argument("--prompt", default="用一句话介绍你自己", help="测试提示词")
    chat_p.set_defaults(func=cmd_chat)

    scan_p = sub.add_parser("scan", help="运行全部探测探针")
    add_common(scan_p)
    scan_p.add_argument("--json", metavar="FILE", help="导出 JSON 报告到文件")
    scan_p.add_argument("--html", metavar="FILE", help="导出 HTML 报告到文件")
    scan_p.add_argument(
        "--context-lengths", metavar="N[,N...]",
        help="长上下文测试档位，逗号分隔（默认 8000），如 8000,32000",
    )
    scan_p.set_defaults(func=cmd_scan)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
