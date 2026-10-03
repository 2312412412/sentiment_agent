"""
数据采集层 - 多源舆情数据获取与预处理
"""
import asyncio
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from bs4 import BeautifulSoup
import aiohttp
import logging

logger = logging.getLogger(__name__)


@dataclass
class RawDocument:
    """原始文档数据结构"""
    content: str
    source: str
    url: str
    title: str
    publish_time: Optional[datetime] = None
    author: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "source": self.source,
            "url": self.url,
            "title": self.title,
            "publish_time": self.publish_time.isoformat() if self.publish_time else None,
            "author": self.author,
            "metadata": self.metadata,
        }


class DataCollector:
    """数据采集器 - 支持多源数据接入"""
    
    def __init__(self, config):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self._seen_urls = set()  # 去重
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def collect_from_sources(
        self,
        company_name: str,
        stock_code: Optional[str] = None,
        sources: List[str] = None,
        date_range: Optional[tuple] = None,
    ) -> List[RawDocument]:
        """
        从多源采集舆情数据
        
        Args:
            company_name: 公司名称
            stock_code: 股票代码
            sources: 数据源列表 [news, announcement, social, forum]
            date_range: 日期范围 (start_date, end_date)
        
        Returns:
            原始文档列表
        """
        sources = sources or ["news", "announcement", "social", "forum"]
        tasks = []
        
        if "news" in sources:
            tasks.append(self._collect_news(company_name, stock_code, date_range))
        if "announcement" in sources:
            tasks.append(self._collect_announcements(company_name, stock_code, date_range))
        if "social" in sources:
            tasks.append(self._collect_social_media(company_name, stock_code, date_range))
        if "forum" in sources:
            tasks.append(self._collect_forums(company_name, stock_code, date_range))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        documents = []
        for result in results:
            if isinstance(result, list):
                documents.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"采集任务失败：{result}")
        
        # 去重
        documents = self._deduplicate(documents)
        logger.info(f"采集完成，共获取 {len(documents)} 篇文档")
        
        return documents
    
    async def _collect_news(
        self, company_name: str, stock_code: Optional[str], date_range: Optional[tuple]
    ) -> List[RawDocument]:
        """采集新闻网站数据"""
        documents = []
        # 示例：可以对接新浪财经、腾讯财经等 API
        # 这里提供框架，实际使用时需要接入具体 API
        logger.info(f"采集新闻数据：{company_name}")
        return documents
    
    async def _collect_announcements(
        self, company_name: str, stock_code: Optional[str], date_range: Optional[tuple]
    ) -> List[RawDocument]:
        """采集交易所公告"""
        documents = []
        # 可以对接巨潮资讯、上交所、深交所公告 API
        logger.info(f"采集公告数据：{company_name}")
        return documents
    
    async def _collect_social_media(
        self, company_name: str, stock_code: Optional[str], date_range: Optional[tuple]
    ) -> List[RawDocument]:
        """采集社交媒体数据"""
        documents = []
        # 可以对接微博、雪球等 API
        logger.info(f"采集社交媒体数据：{company_name}")
        return documents
    
    async def _collect_forums(
        self, company_name: str, stock_code: Optional[str], date_range: Optional[tuple]
    ) -> List[RawDocument]:
        """采集财经论坛数据"""
        documents = []
        # 可以对接东方财富股吧、同花顺论坛等
        logger.info(f"采集论坛数据：{company_name}")
        return documents
    
    def _deduplicate(self, documents: List[RawDocument]) -> List[RawDocument]:
        """基于 URL 和内容哈希去重"""
        seen = set()
        unique_docs = []
        
        for doc in documents:
            # 生成内容哈希
            content_hash = hashlib.md5(doc.content.encode()).hexdigest()
            key = f"{doc.url}:{content_hash}"
            
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)
        
        return unique_docs
    
    async def fetch_url(self, url: str) -> Optional[str]:
        """抓取网页内容"""
        try:
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    return self._clean_html(html)
        except Exception as e:
            logger.error(f"抓取失败 {url}: {e}")
        return None
    
    def _clean_html(self, html: str) -> str:
        """清洗 HTML，提取纯文本"""
        soup = BeautifulSoup(html, "html.parser")
        
        # 移除脚本和样式
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        
        # 提取文本
        text = soup.get_text(separator="\n", strip=True)
        
        # 清理空白行
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines)
    
    def preprocess_text(self, text: str) -> str:
        """文本预处理：去噪、标准化"""
        # 移除多余空白
        text = re.sub(r"\s+", " ", text)
        
        # 移除特殊字符（保留中文标点）
        text = re.sub(r"[^\w\u4e00-\u9fff，。！？；：""''、·…]", " ", text)
        
        # 移除 URL
        text = re.sub(r"http[s]?://\S+", "", text)
        
        return text.strip()


class DataCollectionLayer:
    """数据采集层 - 封装采集逻辑"""
    
    def __init__(self, config):
        self.config = config
        self.collector = DataCollector(config)
    
    async def process(
        self,
        company_name: str,
        stock_code: Optional[str] = None,
        sources: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        执行数据采集流程
        
        Returns:
            预处理后的文档列表
        """
        async with self.collector:
            raw_docs = await self.collector.collect_from_sources(
                company_name=company_name,
                stock_code=stock_code,
                sources=sources,
            )
            
            # 预处理
            processed_docs = []
            for doc in raw_docs:
                doc.content = self.collector.preprocess_text(doc.content)
                processed_docs.append(doc.to_dict())
            
            return processed_docs
