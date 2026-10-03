# 四 Agent 实现文档

## 概述

基于 LangChain 实现了四个核心 Agent，分别对应舆情分析系统的四层架构。

---

## Agent 清单

| Agent | 文件 | 职责 | 核心工具 |
|-------|------|------|----------|
| **DataCollectionAgent** | `agents/data_collector_agent.py` | 多源数据采集 | news_search, entity_recognition |
| **TextAnalysisAgent** | `agents/text_analyst_agent.py` | 文本深度分析 | sentiment_analysis, entity_recognition, event_extraction |
| **DecisionAgent** | `agents/decision_agent.py` | 综合决策分析 | (主要依靠 LLM) |
| **PresentationAgent** | `agents/presentation_agent.py` | 结果展示交互 | (生成报告/图表) |

---

## 架构图

```
用户请求
    ↓
┌─────────────────────────────────────────────────┐
│  SentimentAnalysisSystem (main.py)              │
│  统一入口，编排四个 Agent                         │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  1. DataCollectionAgent                         │
│  - 并行采集多数据源                              │
│  - 新闻/公告/社交媒体/论坛                        │
│  - 去重清洗                                      │
└─────────────────────────────────────────────────┘
    ↓ (原始文档列表)
┌─────────────────────────────────────────────────┐
│  2. TextAnalysisAgent                           │
│  - 批量分析文档                                  │
│  - 情感分析 (带降级)                             │
│  - 实体识别/事件提取                             │
│  - 文本摘要/关键词                               │
└─────────────────────────────────────────────────┘
    ↓ (分析后的文档)
┌─────────────────────────────────────────────────┐
│  3. DecisionAgent                               │
│  - 数据汇总统计                                  │
│  - 规则引擎初判                                  │
│  - LLM 综合分析                                  │
│  - 生成决策报告                                  │
└─────────────────────────────────────────────────┘
    ↓ (DecisionReport)
┌─────────────────────────────────────────────────┐
│  4. PresentationAgent                           │
│  - Markdown/HTML/JSON 报告生成                   │
│  - 图表数据生成                                  │
│  - Gradio 界面交互                               │
└─────────────────────────────────────────────────┘
    ↓
用户查看报告
```

---

## 使用示例

### 基础使用

```python
import asyncio
from langchain_community.chat_models import ChatOpenAI
from sentiment_agent.main import create_sentiment_system
from sentiment_agent.tools import news_search, sentiment_analysis

# 1. 初始化 LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 2. 准备工具
tools = [news_search, sentiment_analysis]

# 3. 创建系统
system = create_sentiment_system(llm, tools)

# 4. 执行分析
result = await system.analyze(
    company_name="贵州茅台",
    stock_code="600519",
    sources=["news", "announcement"],
    date_range=("2026-03-10", "2026-03-17"),
)

# 5. 查看报告
print(result["report"])
```

### 分步使用 Agent

```python
# 创建工厂
from sentiment_agent.agents.agent_factory import AgentFactory

factory = AgentFactory(llm=llm, tools=tools)

# 单独使用数据采集 Agent
collector = factory.create_agent("data_collector")
documents = await collector.execute(
    company_name="贵州茅台",
    stock_code="600519",
)

# 单独使用文本分析 Agent
analyst = factory.create_agent("text_analyst")
analyzed = await analyst.execute(documents=documents)

# 单独使用决策 Agent
decision = factory.create_agent("decision_maker")
report = await decision.execute(
    company_name="贵州茅台",
    stock_code="600519",
    analyzed_documents=analyzed,
)
```

### 启动 Web 界面

```python
# 启动 Gradio 界面
system.launch_gradio(share=False)
# 访问 http://localhost:7860
```

---

## 核心特性

### 1. DataCollectionAgent

- ✅ 并行数据采集 (`asyncio.gather`)
- ✅ 多数据源支持 (新闻/公告/社交媒体/论坛)
- ✅ 智能去重 (基于 URL/标题哈希)
- ✅ 错误容忍 (部分失败不影响整体)

### 2. TextAnalysisAgent

- ✅ 批量处理 (可配置 batch_size)
- ✅ 顺序工具链 (实体→事件→情感→摘要→关键词)
- ✅ 降级处理 (专业工具→LLM→关键词匹配)
- ✅ API 限流保护 (自动延迟)

### 3. DecisionAgent

- ✅ 数据汇总统计
- ✅ 规则引擎初判 (4 条核心规则)
- ✅ LLM 深度分析 (JSON 格式输出)
- ✅ 风险等级评估 (low/medium/high/critical)

### 4. PresentationAgent

- ✅ 多格式报告 (Markdown/HTML/JSON)
- ✅ 图表数据生成 (ECharts/Chart.js 兼容)
- ✅ Gradio 交互界面
- ✅ 风险等级可视化 (🟢🟡🟠🔴)

---

## 配置示例

### config/agents.yaml

```yaml
agents:
  data_collector:
    batch_size: 20
    timeout: 60
    sources:
      - news
      - announcement
    
  text_analyst:
    batch_size: 10
    api_delay: 0.5
    enable_cache: true
    
  decision:
    enable_rules: true
    llm_temperature: 0.3
    max_documents: 10
    
  presentation:
    default_format: markdown
    include_charts: true
```

---

## 依赖安装

```bash
# 核心依赖
pip install langchain langchain-core langchain-community

# 可选：Web 界面
pip install gradio

# 可选：报告导出
pip install markdown pdfkit
```

---

## 测试

### 单元测试

```bash
# 运行测试
pytest tests/agents/
```

### 集成测试

```python
# tests/test_full_pipeline.py
async def test_full_analysis():
    llm = ChatOpenAI(model="gpt-4")
    tools = [news_search, sentiment_analysis]
    system = create_sentiment_system(llm, tools)
    
    result = await system.analyze("贵州茅台", "600519")
    
    assert result["report"] is not None
    assert result["raw_report"] is not None
    assert result["raw_report"].risk_assessment.risk_level in [
        "low", "medium", "high", "critical"
    ]
```

---

## 性能优化建议

1. **启用缓存**: 对重复的公司查询启用结果缓存
2. **批量处理**: 调整 batch_size 平衡速度和 API 限制
3. **并行采集**: 确保数据源采集真正并行执行
4. **异步 I/O**: 所有外部调用使用 async/await

---

## 文件结构

```
sentiment_agent/
├── main.py                          # 主入口，系统编排
├── agents/
│   ├── base_agent.py                # Agent 基类
│   ├── agent_factory.py             # Agent 工厂
│   ├── data_collector_agent.py      # 数据采集 Agent
│   ├── text_analyst_agent.py        # 文本分析 Agent
│   ├── decision_agent.py            # 决策 Agent
│   └── presentation_agent.py        # 展示 Agent
├── tools/
│   └── ...                          # 工具定义
├── layers/
│   └── ...                          # 分层逻辑
└── config/
    └── agents.yaml                  # Agent 配置
```

---

## 下一步

- [ ] 实现完整的工具配置管理
- [ ] 添加 Agent 调用监控
- [ ] 完善 Gradio 界面功能
- [ ] 添加报告导出 (PDF/Word)
- [ ] 实现 Agent 间消息队列

---

**创建时间**: 2026-03-17
**作者**: OpenClaw
