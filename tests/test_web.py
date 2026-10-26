"""Web 仪表盘集成测试（需安装 web 依赖，否则自动跳过）。"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from checker.web.app import app  # noqa: E402


class MockOpenAIHandler(BaseHTTPRequestHandler):
    """返回固定响应的 OpenAI 兼容 mock 服务器。"""

    def do_POST(self):
        body = json.dumps(
            {
                "model": "gpt-4o",
                "choices": [{"message": {"role": "assistant", "content": "15"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def mock_server():
    server = HTTPServer(("127.0.0.1", 0), MockOpenAIHandler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}/v1"
    server.shutdown()


def test_index_page():
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "掺水" in r.text


def test_scan_endpoint(mock_server):
    client = TestClient(app)
    r = client.post(
        "/scan",
        json={
            "provider": "openai",
            "base_url": mock_server,
            "api_key": "sk-test",
            "model": "gpt-4o",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "verdict" in data and "probes" in data
    assert len(data["probes"]) > 0
