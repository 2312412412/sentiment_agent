# 系统架构详解

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        应用展示层 (Gradio)                        │
│  - 用户界面            - 可视化图表          - 报告展示            │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                        总结决策层                                 │
│  - LLM 综合分析         - 风险评估             - 规则引擎          │
│  - 向量检索            - 知识库查询           - 决策建议          │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                        文本分析层                                 │
│  - 实体识别 (NER)       - 事件提取             - 情感分析          │
│  - 文本摘要            - 关键词提取           - 文档分段          │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                        数据采集层                                 │
│  - 新闻网站            - 交易所公告           - 社交媒体          │
│  - 财经论坛            - 数据清洗             - 去重预处理        │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 智能体架构 (ReAct 范式)

每个智能体遵循 `Reason -> Act -> Observe` 循环:

```
┌──────────────┐
│   任务输入    │
└──────┬───────┘
       ↓
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Reason     │────→│    Act       │────→│   Observe    │
│   (推理)     │     │   (行动)     │     │   (观察)     │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       │              ┌─────┴─────┐              │
       │              │ 工具调用   │              │
       │              │ - 新闻搜索 │              │
       │              │ - 情感分析 │              │
       │              │ - 实体识别 │              │
       │              │ - Agent 调用│              │
       │              └───────────┘              │
       │                                         │
       └──────────────────←──────────────────────┘
                    (循环直到完成)
```

## 3. 记忆系统

```
┌─────────────────────────────────────────────┐
│              全局记忆 (Global Memory)         │
│  - 任务进程追踪                              │
│  - 长期目标                                  │
│  - 阶段性成果                                │
│  - 公司缓存                                  │
│  - 持久化存储 (JSON)                         │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│              会话记忆 (Session Memory)        │
│  - 当前对话上下文                            │
│  - 短期缓存 (LRU)                            │
│  - 会话隔离                                  │
└─────────────────────────────────────────────┘
```

## 4. 工具系统

### 内置工具

| 工具 | 功能 | 输入 | 输出 |
|------|------|------|------|
| news_search | 新闻搜索 | query, date_range, limit | 新闻列表 |
| sentiment_analysis | 情感分析 | text | {label, score, aspects} |
| entity_recognition | 实体识别 | text | 实体列表 |
| event_extraction | 事件提取 | text, entities | 事件列表 |
| agent_caller | Agent 调用 | agent_name, task | Agent 结果 |

### 自定义工具

继承 `BaseTool` 创建自定义工具:

```python
from langchain_core.tools import BaseTool

class CustomTool(BaseTool):
    name = "custom_tool"
    description = "工具描述"
    
    def _run(self, arg1: str, arg2: int) -> str:
        # 同步实现
        pass
    
    async def _ainvoke(self, arg1: str, arg2: int) -> str:
        # 异步实现
        pass
```

## 5. 数据流

```
用户请求
   ↓
[数据采集层] 多源数据获取 → 清洗去重 → 标准化文档
   ↓
[文本分析层] 实体识别 → 事件提取 → 情感分析 → 摘要生成
   ↓
[总结决策层] 结果汇总 → 向量检索 → 规则初判 → LLM 综合分析
   ↓
[应用展示层] 报告格式化 → 可视化 → Gradio 界面展示
```

## 6. 扩展点

### 添加新数据源

在 `layers/data_collection.py` 中:

```python
async def _collect_custom_source(self, company_name, stock_code, date_range):
    """自定义数据源采集"""
    documents = []
    # 实现采集逻辑
    return documents
```

### 添加新分析工具

在 `tools/` 目录创建新工具:

```python
@tool
async def custom_analysis(text: str) -> Dict[str, Any]:
    """自定义分析工具"""
    # 实现分析逻辑
    return result
```

### 自定义风险评估规则

在 `layers/decision.py` 中扩展 `rule_based_assessment`:

```python
async def _rule_based_assessment(self, summary_data):
    rules = [
        ("负面新闻 > 5 篇", "high"),
        ("涉及处罚事件", "high"),
        # 添加更多规则
    ]
    # 实现规则引擎
```

## 7. 部署建议

### 开发环境

```bash
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env 填写 API key
python main.py
```

### 生产环境

1. **配置管理**: 使用环境变量或配置中心
2. **日志**: 接入 ELK 或类似系统
3. **监控**: Prometheus + Grafana
4. **缓存**: Redis 缓存热点数据
5. **队列**: Celery + Redis 处理异步任务
6. **数据库**: PostgreSQL 存储历史报告

### Docker 部署

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

## 8. 性能优化

1. **批量处理**: 文档分析使用 batch 处理
2. **缓存**: LLM 响应缓存、公司词典缓存
3. **异步**: 全链路异步 IO
4. **分页**: 大数据集分页处理
5. **限流**: API 调用限流保护
