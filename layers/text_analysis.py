"""
文本分析层 - 语义分析、实体识别、事件提取、情感分析
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


@dataclass
class EntityMention:
    """实体提及"""
    text: str
    entity_type: str  # COMPANY, PERSON, ORG, etc.
    stock_code: Optional[str] = None
    confidence: float = 0.0
    start_pos: int = 0
    end_pos: int = 0


@dataclass
class Event:
    """事件结构"""
    event_type: str  # 融资、并购、处罚、业绩等
    trigger: str  # 触发词
    arguments: Dict[str, Any] = field(default_factory=dict)
    time: Optional[datetime] = None
    confidence: float = 0.0


@dataclass
class SentimentResult:
    """情感分析结果"""
    label: str  # positive, neutral, negative
    score: float
    aspects: Dict[str, float] = field(default_factory=dict)  # 方面级情感


@dataclass
class AnalyzedDocument:
    """分析后的文档"""
    original: Dict[str, Any]
    entities: List[EntityMention] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    sentiment: Optional[SentimentResult] = None
    summary: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original,
            "entities": [
                {
                    "text": e.text,
                    "type": e.entity_type,
                    "stock_code": e.stock_code,
                    "confidence": e.confidence,
                }
                for e in self.entities
            ],
            "events": [
                {
                    "type": ev.event_type,
                    "trigger": ev.trigger,
                    "arguments": ev.arguments,
                    "time": ev.time.isoformat() if ev.time else None,
                }
                for ev in self.events
            ],
            "sentiment": {
                "label": self.sentiment.label,
                "score": self.sentiment.score,
                "aspects": self.sentiment.aspects,
            } if self.sentiment else None,
            "summary": self.summary,
            "keywords": self.keywords,
        }


class TextAnalysisLayer:
    """文本分析层"""
    
    def __init__(self, config, llm, tools: Dict[str, Any] = None):
        self.config = config
        self.llm = llm
        self.tools = tools or {}
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        )
        
        # 上市公司实体词典
        self.company_dict = {}
    
    def load_company_dictionary(self, companies: List[Dict[str, Any]]):
        """
        加载上市公司基础信息库
        
        Args:
            companies: 公司信息列表 [{stock_code, name, aliases, ...}]
        """
        for company in companies:
            stock_code = company.get("stock_code")
            name = company.get("name")
            aliases = company.get("aliases", [])
            
            # 建立名称到公司的映射
            for alias in [name] + aliases:
                self.company_dict[alias] = {
                    "stock_code": stock_code,
                    "name": name,
                }
        
        logger.info(f"加载公司词典，共 {len(self.company_dict)} 个别名")
    
    async def analyze_documents(
        self,
        documents: List[Dict[str, Any]],
        target_company: Optional[str] = None,
    ) -> List[AnalyzedDocument]:
        """
        批量分析文档
        
        Args:
            documents: 原始文档列表
            target_company: 目标公司名称（用于聚焦分析）
        
        Returns:
            分析后的文档列表
        """
        results = []
        
        for doc in documents:
            analyzed = await self._analyze_single(doc, target_company)
            results.append(analyzed)
        
        logger.info(f"完成 {len(results)} 篇文档分析")
        return results
    
    async def _analyze_single(
        self,
        doc: Dict[str, Any],
        target_company: Optional[str],
    ) -> AnalyzedDocument:
        """分析单篇文档"""
        content = doc.get("content", "")
        
        # 1. 实体识别
        entities = await self._recognize_entities(content, target_company)
        
        # 2. 事件提取
        events = await self._extract_events(content, entities)
        
        # 3. 情感分析
        sentiment = await self._analyze_sentiment(content, entities)
        
        # 4. 文本摘要
        summary = await self._summarize(content)
        
        # 5. 关键词提取
        keywords = await self._extract_keywords(content)
        
        return AnalyzedDocument(
            original=doc,
            entities=entities,
            events=events,
            sentiment=sentiment,
            summary=summary,
            keywords=keywords,
        )
    
    async def _recognize_entities(
        self,
        text: str,
        target_company: Optional[str],
    ) -> List[EntityMention]:
        """
        命名实体识别 - 识别上市公司及相关主体
        """
        entities = []
        
        # 基于词典的匹配
        for alias, info in self.company_dict.items():
            if alias in text:
                # 查找所有出现位置
                start = 0
                while True:
                    pos = text.find(alias, start)
                    if pos == -1:
                        break
                    
                    entities.append(EntityMention(
                        text=alias,
                        entity_type="COMPANY",
                        stock_code=info["stock_code"],
                        confidence=0.95,
                        start_pos=pos,
                        end_pos=pos + len(alias),
                    ))
                    start = pos + len(alias)
        
        # 使用 LLM 识别其他实体
        if self.tools.get("entity_recognition"):
            llm_entities = await self.tools["entity_recognition"].ainvoke({"text": text})
            entities.extend(llm_entities)
        
        # 去重
        seen = set()
        unique_entities = []
        for e in entities:
            key = (e.text, e.entity_type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(e)
        
        return unique_entities
    
    async def _extract_events(
        self,
        text: str,
        entities: List[EntityMention],
    ) -> List[Event]:
        """
        事件提取 - 识别舆情相关事件
        """
        events = []
        
        # 定义事件类型和触发词
        event_triggers = {
            "融资": ["融资", "定增", "配股", "发债", "贷款"],
            "并购": ["并购", "收购", "重组", "合并", "收购"],
            "处罚": ["处罚", "罚款", "立案", "调查", "违规"],
            "业绩": ["业绩", "盈利", "亏损", "营收", "净利润"],
            "人事": ["辞职", "任免", "任命", "高管", "董事长"],
            "诉讼": ["诉讼", "起诉", "仲裁", "纠纷", "判决"],
            "合作": ["合作", "签约", "战略", "协议", "合资"],
        }
        
        # 查找触发词
        for event_type, triggers in event_triggers.items():
            for trigger in triggers:
                if trigger in text:
                    events.append(Event(
                        event_type=event_type,
                        trigger=trigger,
                        confidence=0.8,
                    ))
        
        # 使用 LLM 进行更深入的事件提取
        if self.tools.get("event_extraction"):
            llm_events = await self.tools["event_extraction"].ainvoke({
                "text": text,
                "entities": [e.text for e in entities],
            })
            events.extend(llm_events)
        
        return events
    
    async def _analyze_sentiment(
        self,
        text: str,
        entities: List[EntityMention],
    ) -> SentimentResult:
        """
        情感分析
        """
        # 使用情感分析工具
        if self.tools.get("sentiment_analysis"):
            result = await self.tools["sentiment_analysis"].ainvoke({"text": text})
            return SentimentResult(
                label=result.get("label", "neutral"),
                score=result.get("score", 0.5),
                aspects=result.get("aspects", {}),
            )
        
        #  fallback: 使用 LLM 判断
        from langchain_core.prompts import ChatPromptTemplate
        
        prompt = ChatPromptTemplate.from_template(
            "分析以下文本的情感倾向，只返回 JSON：{{\"label\": \"positive/neutral/negative\", \"score\": 0-1}}\n\n文本：{text}"
        )
        chain = prompt | self.llm
        
        response = await chain.ainvoke({"text": text[:2000]})  # 限制长度
        
        # 解析响应（简化处理）
        return SentimentResult(
            label="neutral",
            score=0.5,
        )
    
    async def _summarize(self, text: str) -> str:
        """文本摘要"""
        if len(text) < 500:
            return text
        
        from langchain_core.prompts import ChatPromptTemplate
        
        prompt = ChatPromptTemplate.from_template(
            "用一句话总结以下文本的核心内容：\n\n{text}"
        )
        chain = prompt | self.llm
        
        response = await chain.ainvoke({"text": text[:3000]})
        return response.content.strip()
    
    async def _extract_keywords(self, text: str) -> List[str]:
        """关键词提取"""
        # 简单实现：可以使用 TF-IDF 或 TextRank
        # 这里使用 LLM 提取
        from langchain_core.prompts import ChatPromptTemplate
        
        prompt = ChatPromptTemplate.from_template(
            "从以下文本中提取 5-10 个关键词，用逗号分隔：\n\n{text}"
        )
        chain = prompt | self.llm
        
        response = await chain.ainvoke({"text": text[:2000]})
        keywords = response.content.strip().split(",")
        return [k.strip() for k in keywords if k.strip()]
    
    def split_document(self, text: str) -> List[str]:
        """文档分段"""
        return self.text_splitter.split_text(text)
