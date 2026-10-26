"""Web 仪表盘：FastAPI 应用。

启动：uvicorn checker.web.app:app --reload
"""
from __future__ import annotations

from pydantic import BaseModel

from ..adapters import OpenAIAdapter, AnthropicAdapter
from ..runner import run_all
from ..scoring.verdict import weigh
from ..report import to_json


class ScanRequest(BaseModel):
    provider: str = "openai"
    base_url: str
    api_key: str
    model: str


def build_adapter(req: ScanRequest):
    if req.provider == "anthropic":
        return AnthropicAdapter(req.base_url, req.api_key, req.model)
    return OpenAIAdapter(req.base_url, req.api_key, req.model)


INDEX_HTML = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>AI 中转站掺水检测器</title>
<style>
body{font-family:sans-serif;max-width:860px;margin:40px auto;padding:0 20px;color:#222;background:#fafafa}
h1{font-size:1.5em}
form{background:#fff;padding:20px;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.1)}
label{display:block;margin:10px 0;font-size:.9em;color:#555}
input,select{width:100%;box-sizing:border-box;padding:9px;margin-top:4px;border:1px solid #ccc;border-radius:6px;font-size:1em}
button{width:100%;padding:12px;margin-top:16px;background:#2563eb;color:#fff;border:0;border-radius:6px;font-size:1em;cursor:pointer}
button:hover{background:#1d4ed8}
.verdict{margin-top:24px;background:#fff;padding:24px;border-radius:8px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.1)}
.score{font-size:3em;font-weight:bold}
.level{font-size:1.1em;margin:8px 0}
table{border-collapse:collapse;width:100%;margin-top:20px;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.1)}
td,th{border:1px solid #eee;padding:9px 12px;text-align:left}
th{background:#f5f5f5}
</style></head><body>
<h1>AI 中转站掺水检测器</h1>
<form id="form">
  <label>协议
    <select name="provider">
      <option value="openai">OpenAI 兼容</option>
      <option value="anthropic">Anthropic 兼容</option>
    </select>
  </label>
  <label>Base URL<input name="base_url" placeholder="https://中转站地址/v1"></label>
  <label>API Key<input name="api_key" type="password"></label>
  <label>声称的模型名<input name="model" placeholder="gpt-4o"></label>
  <button type="submit">开始检测</button>
</form>
<div id="result"></div>
<script>
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
document.getElementById('form').addEventListener('submit', async e => {
  e.preventDefault();
  const data = Object.fromEntries(new FormData(e.target));
  const box = document.getElementById('result');
  box.innerHTML = '<div class="verdict">检测中，请稍候…</div>';
  try {
    const resp = await fetch('/scan', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(data),
    });
    if (!resp.ok) throw new Error((await resp.json()).detail || resp.statusText);
    const j = await resp.json();
    const v = j.verdict;
    const color = v.score >= 0.6 ? '#d32f2f' : (v.score >= 0.3 ? '#f57c00' : '#2e7d32');
    const rows = j.probes.map(p =>
      `<tr><td>${esc(p.probe)}</td><td>${p.score.toFixed(2)}</td><td>${esc(p.verdict)}</td></tr>`
    ).join('');
    box.innerHTML = `
      <div class="verdict">
        <div class="score" style="color:${color}">${Math.round(v.score * 100)}%</div>
        <div class="level">${esc(v.level)}</div>
        <div>替换维度 ${Math.round(v.substitution * 100)}% · 降智维度 ${Math.round(v.degradation * 100)}%</div>
      </div>
      <table><tr><th>探针</th><th>可疑度</th><th>结论</th></tr>${rows}</table>`;
  } catch (err) {
    box.innerHTML = `<div class="verdict"><div class="level" style="color:#d32f2f">检测失败：${esc(err.message)}</div></div>`;
  }
});
</script>
</body></html>"""


def create_app():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="AI 中转站掺水检测器")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML

    @app.post("/scan")
    def scan(req: ScanRequest):
        adapter = build_adapter(req)
        results = run_all(adapter)
        verdict = weigh(results)
        return to_json(results, verdict, req.model, req.base_url)

    return app


app = create_app()
