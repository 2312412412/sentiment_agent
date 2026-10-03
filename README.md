# 上市公司舆情分析智能体系统

基于 LangChain 的多智能体舆情分析系统，面向上市公司新闻、公告和社交媒体文本，提供数据采集、实体识别、情感分析、事件提取、风险评估和 Gradio 可视化报告。

> 说明：本项目输出仅用于舆情研究和辅助分析，不构成投资建议。

## 核心能力

- 多源采集：通过 Tavily 搜索、网页抓取和上市公司基础信息工具收集文本线索。
- 文本分析：识别公司/人物/机构/股票代码实体，抽取处罚、诉讼、业绩、融资等事件信号。
- 决策评估：结合规则评分和 LLM 生成结构化风险报告。
- 可视化界面：内置 Gradio 页面，支持 Markdown、HTML、JSON 报告输出。
- 记忆机制：将历史公司分析记录写入本地 `memory_data/`，用于后续分析参考。

## 项目结构

```text
sentiment_agent/
├── agents/                  # 四类核心 Agent
├── layers/                  # 分层实现与早期模块
├── memory/                  # 会话记忆和全局记忆
├── skills/                  # 可复用的数据采集技能
├── tools/                   # 搜索、爬取、情感、实体、事件工具
├── config.py                # 环境变量配置
├── main.py                  # 系统入口与 Gradio 启动
├── example_usage.py         # 编程调用示例
├── requirements.txt         # Python 依赖
└── .env.example             # 环境变量模板
```

## 环境要求

- Python 3.10+
- 可访问的 LLM API Key
- 可选：Tavily API Key，用于更稳定的新闻/网页搜索

## 快速开始

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`，至少填写：

```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
LLM_API_KEY=your_api_key_here
GRADIO_SHARE=false
```

启动 Web 界面：

```bash
python main.py
```

默认访问地址为 `http://127.0.0.1:7860`。

## 编程调用

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
        output_format="markdown",
    )
    print(result["report"])


asyncio.run(run())
```

## 配置项

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LLM_PROVIDER` | `openai` | LLM 提供方，支持 `openai` / `dashscope` |
| `LLM_MODEL` | `gpt-4o` | 使用的模型名称 |
| `LLM_API_KEY` | 空 | 必填，LLM API Key |
| `LLM_BASE_URL` | 按 provider 推断 | 可选，自定义兼容 OpenAI 的接口地址 |
| `TAVILY_API_KEY` | 空 | 可选，缺失时搜索工具会返回空结果 |
| `GRADIO_PORT` | `7860` | Gradio 本地端口 |
| `GRADIO_SHARE` | `false` | 是否开启 Gradio 临时公网分享链接 |
| `MEMORY_PATH` | `./memory_data` | 本地记忆数据目录 |
| `SENTIMENT_THRESHOLD` | `0.6` | 情感阈值配置 |

DashScope 示例：

```bash
LLM_PROVIDER=dashscope
LLM_MODEL=qwen-max
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_API_KEY=your_dashscope_key
```

## 开源发布前检查

- 不要提交 `.env`、证书、缓存、运行时记忆和模型文件。
- 如果 API Key 曾经进入 Git 历史，请立即在服务商后台撤销并重新生成。
- 发布前请选择许可证并添加 `LICENSE` 文件；没有许可证时，GitHub 上的代码默认保留全部权利。
- 对外展示报告前，请确认报告中不包含未授权的第三方内容、个人信息或敏感业务数据。

## 更多文档

- [快速启动指南](QUICKSTART.md)
- [系统架构详解](ARCHITECTURE.md)
- [四 Agent 实现文档](AGENTS_IMPLEMENTATION.md)
- [API/Tools 申请清单](API_APPLICATION_LIST.md)
