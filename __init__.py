"""
舆情分析智能体系统

基于 LangChain 的多层多智能体架构，支持:
- 多源数据接入
- 上市公司实体识别
- 情感分析
- 风险评估
- Gradio 可视化界面

Usage:
    from sentiment_agent import build_default_system, config
    
    system = build_default_system(config)
    result = await system.analyze("贵州茅台", "600519")
"""

__version__ = "1.0.0"
__author__ = "OpenClaw Agent"

from .main import SentimentAnalysisSystem, build_default_system, create_sentiment_system
from .config import Config, config

__all__ = [
    "SentimentAnalysisSystem",
    "build_default_system",
    "create_sentiment_system",
    "Config",
    "config",
]
