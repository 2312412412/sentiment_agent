"""
总结决策层 - 整合分析、风险评估、决策建议
"""
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger(__name__)


@dataclass
class RiskAssessment:
    """风险评估结果"""
    risk_level: str  # low, medium, high, critical
    risk_score: float  # 0-1
    risk_factors: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class DecisionReport:
    """决策报告"""
    company_name: str
    stock_code: Optional[str]
    analysis_period: str
    overall_sentiment: str
    risk_assessment: RiskAssessment
    key_events: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    summary: str = ""
    detailed_analysis: str = ""
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "company_name": self.company_name,
            "stock_code": self.stock_code,
            "analysis_period": self.analysis_period,
            "overall_sentiment": self.overall_sentiment,
            "risk_assessment": {
                "level": self.risk_assessment.risk_level,
                "score": self.risk_assessment.risk_score,
                "factors": self.risk_assessment.risk_factors,
                "confidence": self.risk_assessment.confidence,
                "reasoning": self.risk_assessment.reasoning,
            },
            "key_events": self.key_events,
            "recommendations": self.recommendations,
            "summary": self.summary,
            "detailed_analysis": self.detailed_analysis,
            "generated_at": self.generated_at.isoformat(),
        }


class DecisionLayer:
    """总结决策层"""
    
    def __init__(self, config, llm, vector_store=None, knowledge_base=None, rule_engine=None):
        self.config = config
        self.llm = llm
        self.vector_store = vector_store
        self.knowledge_base = knowledge_base
        self.rule_engine = rule_engine
        
        self._setup_prompts()
    
    def _setup_prompts(self):
        """设置提示模板"""
        
        # 综合分析提示
        self.analysis_prompt = ChatPromptTemplate.from_template(
            """你是专业的上市公司舆情分析专家。请基于以下分析结果进行综合评估。

## 分析数据
- 公司：{company_name} ({stock_code})
- 分析文档数：{doc_count}
- 时间范围：{time_range}

## 情感分布
{sentiment_summary}

## 关键事件
{events_summary}

## 实体提及
{entities_summary}

## 任务
1. 评估整体舆情风险等级（low/medium/high/critical）
2. 识别主要风险因素
3. 给出投资建议或应对措施
4. 生成简明摘要

请以 JSON 格式输出：
{{
    "risk_level": "...",
    "risk_score": 0.0-1.0,
    "risk_factors": [...],
    "recommendations": [...],
    "summary": "...",
    "detailed_analysis": "..."
}}
"""
        )
        
        # 风险判定提示
        self.risk_prompt = ChatPromptTemplate.from_template(
            """基于以下规则评估舆情风险：

{rules}

## 当前情况
{context}

请评估风险等级并说明理由。
"""
        )
        
        self.json_parser = JsonOutputParser()
    
    async def generate_report(
        self,
        company_name: str,
        stock_code: Optional[str],
        analyzed_documents: List[Dict[str, Any]],
        time_range: Optional[str] = None,
    ) -> DecisionReport:
        """
        生成综合舆情分析报告
        
        Args:
            company_name: 公司名称
            stock_code: 股票代码
            analyzed_documents: 分析后的文档列表
            time_range: 时间范围描述
        
        Returns:
            决策报告
        """
        logger.info(f"生成报告：{company_name}")
        
        # 1. 汇总分析结果
        summary_data = self._aggregate_results(analyzed_documents)
        
        # 2. 向量检索相似历史案例
        similar_cases = await self._retrieve_similar_cases(summary_data)
        
        # 3. 规则引擎初判
        preliminary_risk = await self._rule_based_assessment(summary_data)
        
        # 4. LLM 综合分析
        llm_analysis = await self._llm_analysis(
            company_name, stock_code, summary_data, preliminary_risk
        )
        
        # 5. 整合报告
        report = DecisionReport(
            company_name=company_name,
            stock_code=stock_code,
            analysis_period=time_range or "最近 7 天",
            overall_sentiment=summary_data["overall_sentiment"],
            risk_assessment=RiskAssessment(
                risk_level=llm_analysis.get("risk_level", "medium"),
                risk_score=llm_analysis.get("risk_score", 0.5),
                risk_factors=llm_analysis.get("risk_factors", []),
                confidence=0.8,
                reasoning=llm_analysis.get("detailed_analysis", ""),
            ),
            key_events=summary_data["key_events"],
            recommendations=llm_analysis.get("recommendations", []),
            summary=llm_analysis.get("summary", ""),
            detailed_analysis=llm_analysis.get("detailed_analysis", ""),
        )
        
        logger.info(f"报告生成完成，风险等级：{report.risk_assessment.risk_level}")
        return report
    
    def _aggregate_results(
        self, documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """汇总分析结果"""
        if not documents:
            return self._empty_summary()
        
        # 情感统计
        sentiments = [
            doc.get("sentiment", {})
            for doc in documents
            if doc.get("sentiment")
        ]
        
        positive_count = sum(1 for s in sentiments if s.get("label") == "positive")
        negative_count = sum(1 for s in sentiments if s.get("label") == "negative")
        neutral_count = len(sentiments) - positive_count - negative_count
        
        overall = "neutral"
        if positive_count > negative_count * 1.5:
            overall = "positive"
        elif negative_count > positive_count * 1.5:
            overall = "negative"
        
        # 事件汇总
        all_events = []
        for doc in documents:
            all_events.extend(doc.get("events", []))
        
        # 实体汇总
        all_entities = []
        for doc in documents:
            all_entities.extend(doc.get("entities", []))
        
        return {
            "doc_count": len(documents),
            "overall_sentiment": overall,
            "sentiment_distribution": {
                "positive": positive_count,
                "neutral": neutral_count,
                "negative": negative_count,
            },
            "key_events": all_events[:10],  # 取前 10 个
            "entities": all_entities[:20],
        }
    
    def _empty_summary(self) -> Dict[str, Any]:
        """空数据汇总"""
        return {
            "doc_count": 0,
            "overall_sentiment": "neutral",
            "sentiment_distribution": {"positive": 0, "neutral": 0, "negative": 0},
            "key_events": [],
            "entities": [],
        }
    
    async def _retrieve_similar_cases(
        self, summary_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """向量检索相似历史案例"""
        if not self.vector_store:
            return []
        
        # 构建查询
        query = f"{summary_data.get('overall_sentiment', '')} 舆情事件"
        
        try:
            results = await self.vector_store.asimilarity_search(query, k=3)
            return [doc.metadata for doc in results]
        except Exception as e:
            logger.error(f"向量检索失败：{e}")
            return []
    
    async def _rule_based_assessment(
        self, summary_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """基于规则的风险初判"""
        if not self.rule_engine:
            return {"risk_level": "medium", "factors": []}
        
        return await self.rule_engine.evaluate(summary_data)
    
    async def _llm_analysis(
        self,
        company_name: str,
        stock_code: Optional[str],
        summary_data: Dict[str, Any],
        preliminary_risk: Dict[str, Any],
    ) -> Dict[str, Any]:
        """LLM 综合分析"""
        # 准备上下文
        sentiment_summary = (
            f"正面：{summary_data['sentiment_distribution']['positive']}篇，"
            f"中性：{summary_data['sentiment_distribution']['neutral']}篇，"
            f"负面：{summary_data['sentiment_distribution']['negative']}篇"
        )
        
        events_summary = "\n".join(
            [f"- {e.get('type', '未知')}: {e.get('trigger', '')}" 
             for e in summary_data["key_events"][:5]]
        ) or "无重大事件"
        
        entities_summary = ", ".join(
            set(e.get("text", "") for e in summary_data["entities"][:10])
        ) or "无"
        
        # 构建提示
        prompt = self.analysis_prompt.format(
            company_name=company_name,
            stock_code=stock_code or "未知",
            doc_count=summary_data["doc_count"],
            time_range="最近 7 天",
            sentiment_summary=sentiment_summary,
            events_summary=events_summary,
            entities_summary=entities_summary,
        )
        
        # 调用 LLM
        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content
            
            # 尝试解析 JSON
            if "{" in content and "}" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end]
                return json.loads(json_str)
        except Exception as e:
            logger.error(f"LLM 分析失败：{e}")
        
        # 默认返回
        return {
            "risk_level": preliminary_risk.get("risk_level", "medium"),
            "risk_score": 0.5,
            "risk_factors": preliminary_risk.get("factors", []),
            "recommendations": ["持续关注舆情动态"],
            "summary": f"对{company_name}的舆情分析完成。",
            "detailed_analysis": "分析完成。",
        }
    
    async def query_knowledge_base(
        self, query: str, company_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """查询领域知识库"""
        if not self.knowledge_base:
            return []
        
        return await self.knowledge_base.query(query, company_name)
