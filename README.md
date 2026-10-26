# AI 中转站「掺水」检测器

黑盒检测 AI API 中转站是否「掺水」：把便宜国产模型冒充 Claude/GPT、同族降级、换量化/蒸馏模型降智、概率性混入廉价模型。无需官方 API 账号，纯黑盒探测，输出 0~100% 掺水概率。

## 安装

```bash
pip install -e .            # 核心（仅 httpx）
pip install -e ".[web]"     # 加 Web 仪表盘（fastapi + uvicorn）
pip install -e ".[dev]"     # 加测试（pytest）
```

## 用法

### CLI 单条测试

```bash
checker chat --provider openai --base-url https://你的中转站/v1 --api-key sk-xxx --model gpt-4o --prompt "你好"
```

### CLI 完整检测

```bash
checker scan --provider openai --base-url https://你的中转站/v1 --api-key sk-xxx --model gpt-4o --json report.json --html report.html
```

支持 `--provider openai|anthropic` 两种协议。`--json` / `--html` 可选导出报告。

### Web 仪表盘

```bash
pip install -e ".[web]"
uvicorn checker.web.app:app --reload
```

浏览器打开 http://127.0.0.1:8000 ，填入中转站地址和 key 即可在线检测。

## 检测原理（三层黑盒探测，全部无需官方基线）

1. **静态 / 协议探测**：字符计数陷阱、知识截止点、延迟画像 —— 识别模型家族与是否量化
2. **行为指纹探测**：随机数输出分布、写作风格、拒绝模式、身份一致性 —— 找模型替换的家族矛盾
3. **能力梯度探测**：数学/代码/推理梯度题库 + 长上下文窗口 —— 判「降智」：简单题会、难题崩、长文失忆

最后按权重融合所有信号，输出掺水概率与结论分类（未发现明显掺水 / 可疑 / 疑似掺水），并细分「模型替换」与「降智」两个维度。

## 项目结构

```
src/checker/
├── adapters/        # OpenAI / Anthropic 协议适配器
├── probes/          # 探测探针（protocol / fingerprint / capability）
├── scoring/         # 统计量 + 加权评分融合
├── fingerprints/    # 指纹库（预留）
├── web/             # FastAPI 仪表盘
├── cli.py           # CLI 入口
├── runner.py        # 探针编排
└── report.py        # 终端 / JSON / HTML 报告
```

## 注意事项

- **Windows 终端中文乱码**：运行前执行 `chcp 65001` 切换到 UTF-8。
- **单次检测有误判可能**：价值在于多次采样看统计趋势，探针的分数与权重可后续调优。
- **本工具只做正向检测**：不涉及任何绕过/攻击手段。
- **局限**：无官方 API 时，无法 100% 证明「就是原厂模型」，只能给出基于多信号交叉验证的掺水概率。
