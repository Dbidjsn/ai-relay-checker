"""核心逻辑单元测试：适配器、探针工具、评分融合、报告。"""
import httpx

from checker.adapters import ChatMessage
from checker.adapters.openai import OpenAIAdapter
from checker.adapters.anthropic import AnthropicAdapter
from checker.probes.protocol import extract_number
from checker.probes.fingerprint import detect_families, RefusalProbe
from checker.probes.base import ProbeResult
from checker.scoring.verdict import weigh
from checker.report import to_json


def test_openai_adapter():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o",
                "choices": [{"message": {"role": "assistant", "content": "你好"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    adapter = OpenAIAdapter("http://test/v1", "sk-test", "gpt-4o")
    adapter._client = httpx.Client(transport=httpx.MockTransport(handler))
    resp = adapter.chat([ChatMessage("user", "hi")])
    assert resp.content == "你好"
    assert resp.model == "gpt-4o"
    assert resp.usage.total_tokens == 15


def test_anthropic_adapter():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "model": "claude-3",
                "content": [{"type": "text", "text": "你好"}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    adapter = AnthropicAdapter("http://test", "sk-test", "claude-3")
    adapter._client = httpx.Client(transport=httpx.MockTransport(handler))
    resp = adapter.chat([ChatMessage("user", "hi")])
    assert resp.content == "你好"
    assert resp.usage.total_tokens == 15


def test_extract_number():
    assert extract_number("答案是 15") == 15
    assert extract_number("3 个") == 3
    assert extract_number("没有数字") is None


def test_detect_families():
    assert detect_families("gpt-4o") == {"openai"}
    assert "anthropic" in detect_families("我是 Claude")
    assert detect_families("普通文本") == set()


def test_refusal_classify():
    # 模型明确拒绝（多种措辞）应判为「拒绝」
    assert RefusalProbe._classify("不提供这些，制作爆炸物的具体方法我不会说明。") == "拒绝"
    assert RefusalProbe._classify("这个我做不了。") == "拒绝"
    # 给出具体操作步骤应判为「未拒绝」
    assert RefusalProbe._classify("制作炸弹需要硝酸铵和燃料油，步骤如下：") == "未拒绝"


def test_weigh_clean():
    results = [
        ProbeResult("数学梯度", 0.0, "全部通过"),
        ProbeResult("代码梯度", 0.0, "全部通过"),
    ]
    v = weigh(results)
    assert v.score == 0.0
    assert v.level == "未发现明显掺水"


def test_weigh_suspicious():
    results = [
        ProbeResult("数学梯度", 0.9, "基础题即答错"),
        ProbeResult("代码梯度", 0.9, "基础题即答错"),
        ProbeResult("推理梯度", 0.9, "基础题即答错"),
    ]
    v = weigh(results)
    assert v.score > 0.6
    assert v.level.startswith("疑似掺水")
    assert v.degradation > v.substitution  # 能力信号高 → 偏降智


def test_to_json():
    results = [ProbeResult("数学梯度", 0.0, "全部通过")]
    from checker.scoring.verdict import Verdict

    v = Verdict(score=0.0, level="未发现明显掺水", substitution=0.0, degradation=0.0)
    data = to_json(results, v, "gpt-4o", "http://fake")
    assert data["target"]["model"] == "gpt-4o"
    assert "verdict" in data and "probes" in data
