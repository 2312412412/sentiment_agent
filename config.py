"""
系统配置管理
"""
import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass
class Config:
    """系统配置"""
    
    # LLM 配置
    llm_provider: str = "openai"  # openai, dashscope, etc.
    llm_model: str = "gpt-4o"
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    
    # 情感分析模型
    sentiment_model_path: str = "./models/sentiment"
    sentiment_threshold: float = 0.6
    
    # 向量检索
    vector_store_path: str = "./vector_store"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # 数据采集
    crawl_delay: float = 1.0  # 爬虫延迟（秒）
    max_pages: int = 100  # 最大爬取页数
    
    # 记忆系统
    memory_path: str = "./memory_data"
    session_ttl: int = 3600  # 会话 TTL（秒）
    
    # Gradio
    gradio_port: int = 7860
    gradio_share: bool = False  # 默认只开启本地访问，避免意外暴露服务
    
    # 搜索
    tavily_api_key: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量加载配置"""
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", cls.llm_provider),
            llm_model=os.getenv("LLM_MODEL", cls.llm_model),
            llm_api_key=os.getenv("LLM_API_KEY"),
            llm_base_url=os.getenv("LLM_BASE_URL"),
            sentiment_model_path=os.getenv("SENTIMENT_MODEL_PATH", cls.sentiment_model_path),
            sentiment_threshold=_get_float("SENTIMENT_THRESHOLD", cls.sentiment_threshold),
            vector_store_path=os.getenv("VECTOR_STORE_PATH", cls.vector_store_path),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            crawl_delay=_get_float("CRAWL_DELAY", cls.crawl_delay),
            max_pages=_get_int("MAX_PAGES", cls.max_pages),
            memory_path=os.getenv("MEMORY_PATH", cls.memory_path),
            session_ttl=_get_int("SESSION_TTL", cls.session_ttl),
            gradio_port=_get_int("GRADIO_PORT", cls.gradio_port),
            gradio_share=_get_bool("GRADIO_SHARE", cls.gradio_share),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
        )


# 全局配置实例
config = Config.from_env()
