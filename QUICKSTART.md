# 快速启动指南

## 1. 环境准备

系统要求：

- Python 3.10+
- 内存 4GB+
- 可访问的 LLM API Key

安装依赖：

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

配置环境变量：

```bash
cp .env.example .env
```

编辑 `.env`，填写至少一个 LLM API Key：

```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
LLM_API_KEY=your_api_key_here
GRADIO_SHARE=false
```

如需使用 Tavily 搜索，再填写：

```bash
TAVILY_API_KEY=your_tavily_key_here
```

## 2. 启动 Web 界面

```bash
python main.py
```

浏览器访问：

```text
http://127.0.0.1:7860
```

默认不会创建公网分享链接。如需临时分享，可在 `.env` 中显式设置：

```bash
GRADIO_SHARE=true
```

## 3. 编程调用

```python
import asyncio

from config import config
from main import build_default_system


async def run():
    system = build_default_system(config)
    result = await system.analyze(
        company_name="贵州茅台",
        stock_code="600519",
        sources=["news", "announcement"],
        date_range=("2026-03-01", "2026-03-07"),
        max_effective_documents=5,
    )
    print(result["report"])


asyncio.run(run())
```

## 4. 常见问题

### `LLM_API_KEY 未配置`

复制 `.env.example` 为 `.env`，并填写 `LLM_API_KEY`。

### 搜索结果为空

检查是否已配置 `TAVILY_API_KEY`。没有该 Key 时，系统仍可运行，但新闻搜索工具会返回空列表。

### Gradio 分享链接失败

Windows 环境下 Gradio share 依赖额外的 `frpc` 二进制文件。本项目默认 `GRADIO_SHARE=false`，建议优先使用本地访问或 SSH 隧道。

### 如何调整分析数量

在 Web 界面填写 `Effective Text Limit`，或在代码中传入 `max_effective_documents`。

## 5. 下一步

- 阅读 `ARCHITECTURE.md` 了解系统分层。
- 查看 `example_usage.py` 了解更多编程调用方式。
- 发布到 GitHub 前确认 `.env`、`memory_data/`、`__pycache__/` 不在提交内容中。
