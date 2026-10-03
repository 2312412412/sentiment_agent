"""
Decision agent for listed-company sentiment analysis.

Responsibilities:
- aggregate analyzed documents
- perform rule-based risk scoring
- optionally refine with LLM
- produce a structured decision report for presentation
"""
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

EVENT_TYPE_ALIASES = {
    "penalty": ["penalty", "\u5904\u7f5a"],
    "litigation": ["litigation", "\u8bc9\u8bbc"],
    "investigation": ["investigation", "\u8c03\u67e5"],
    "violation": ["violation", "\u8fdd\u89c4"],
    "case_filing": ["case_filing", "\u7acb\u6848"],
    "earnings": ["earnings", "\u4e1a\u7ee9"],
    "financing": ["financing", "\u878d\u8d44"],
    "merger": ["merger", "\u5e76\u8d2d"],
    "cooperation": ["cooperation", "\u5408\u4f5c"],
    "personnel": ["personnel", "\u4eba\u4e8b"],
    "unknown": ["unknown"],
}
EVENT_TYPE_LOOKUP = {
    alias.lower(): canonical
    for canonical, aliases in EVENT_TYPE_ALIASES.items()
    for alias in aliases
}
EVENT_TYPE_DISPLAY = {
    "penalty": "Penalty",
    "litigation": "Litigation",
    "investigation": "Investigation",
    "violation": "Violation",
    "case_filing": "Case Filing",
    "earnings": "Earnings",
    "financing": "Financing",
    "merger": "Merger",
    "cooperation": "Cooperation",
    "personnel": "Personnel",
    "unknown": "Unknown",
}
HIGH_RISK_EVENT_TYPES = {
    "penalty",
    "litigation",
    "investigation",
    "violation",
    "case_filing",
}
EVENT_TYPE_RISK_WEIGHTS = {
    "penalty": 1.0,
    "litigation": 0.95,
    "investigation": 0.95,
    "violation": 0.9,
    "case_filing": 0.9,
    "earnings": 0.65,
    "financing": 0.6,
    "merger": 0.55,
    "cooperation": 0.45,
}


@dataclass
class RiskAssessment:
    """Risk assessment result."""

    risk_level: str  # low, medium, high, critical
    risk_score: float  # 0-1
    risk_factors: List[str] = field(default_factory=list)
    confidence: float = 0.8
    reasoning: str = ""


@dataclass
class DecisionReport:
    """Final decision report."""

    company_name: str
    stock_code: Optional[str]
    analysis_period: str
    overall_sentiment: Dict[str, Any]
    risk_assessment: RiskAssessment
    key_events: List[Dict[str, Any]]
    recommendations: List[str]
    summary: str
    detailed_analysis: str = ""
    sentiment_distribution: Dict[str, int] = field(default_factory=dict)
    sentiment_timeline: List[Dict[str, Any]] = field(default_factory=list)
    event_counts: Dict[str, int] = field(default_factory=dict)
    negative_event_summaries: List[str] = field(default_factory=list)
    data_overview: Dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "company_name": self.company_name,
            "stock_code": self.stock_code,
            "analysis_period": self.analysis_period,
            "overall_sentiment": self.overall_sentiment,
            "risk_assessment": {
                "risk_level": self.risk_assessment.risk_level,
                "risk_score": self.risk_assessment.risk_score,
                "risk_factors": self.risk_assessment.risk_factors,
                "confidence": self.risk_assessment.confidence,
                "reasoning": self.risk_assessment.reasoning,
            },
            "key_events": self.key_events,
            "recommendations": self.recommendations,
            "summary": self.summary,
            "detailed_analysis": self.detailed_analysis,
            "sentiment_distribution": self.sentiment_distribution,
            "sentiment_timeline": self.sentiment_timeline,
            "event_counts": self.event_counts,
            "negative_event_summaries": self.negative_event_summaries,
            "data_overview": self.data_overview,
            "generated_at": self.generated_at,
        }


class DecisionAgent(BaseAgent):
    """
    Decision layer agent.

    It blends explicit rules and (optional) LLM refinement.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[Any] = None,
        config: Optional[Any] = None,
    ):
        super().__init__(
            name="decision_maker",
            llm=llm,
            tools=tools or [],
            memory=memory,
            config=config,
        )
        self.rules = self._load_rules()

    def _load_rules(self) -> List[Dict[str, Any]]:
        return [
            {"name": "negative_ratio_high", "threshold": 0.5, "weight": 0.22},
            {"name": "negative_ratio_mid", "threshold": 0.3, "weight": 0.12},
            {
                "name": "high_risk_event",
                "event_types": ["penalty", "litigation", "investigation", "violation", "case_filing"],
                "weight": 0.16,
            },
            {"name": "event_volume_spike", "threshold": 10, "weight": 0.1},
            {"name": "trend_deteriorating", "weight": 0.14},
            {"name": "historical_deterioration", "weight": 0.12},
        ]

    async def execute(
        self,
        company_name: str,
        stock_code: Optional[str],
        analyzed_documents: List[Dict[str, Any]],
        time_range: Optional[str] = None,
        memory_context: Optional[Dict[str, Any]] = None,
        analysis_summary: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> DecisionReport:
        self.state.task = f"analyze sentiment risk for {company_name}"
        self.state.status = "running"

        summary_data = self._aggregate_results(
            analyzed_documents=analyzed_documents,
            analysis_summary=analysis_summary,
        )
        preliminary_risk = await self._rule_based_assessment(
            summary_data=summary_data,
            memory_context=memory_context or {},
        )
        llm_analysis = await self._llm_analysis(
            company_name=company_name,
            stock_code=stock_code,
            summary_data=summary_data,
            preliminary_risk=preliminary_risk,
            memory_context=memory_context or {},
            documents=analyzed_documents[:8],
        )

        llm_score = self._safe_float(llm_analysis.get("risk_score"), default=None)
        if llm_score is None:
            final_score = preliminary_risk["risk_score"]
            confidence = 0.75
        else:
            final_score = round(min(1.0, max(0.0, preliminary_risk["risk_score"] * 0.65 + llm_score * 0.35)), 2)
            confidence = round(min(1.0, max(0.0, 0.72 + abs(llm_score - preliminary_risk["risk_score"]) * 0.1)), 2)

        risk_level = self._score_to_level(final_score)
        llm_factors = self._safe_str_list(llm_analysis.get("risk_factors"))
        combined_factors = self._merge_text_lists(
            preliminary_risk.get("risk_factors", []),
            llm_factors,
        )
        recommendations = self._safe_str_list(llm_analysis.get("recommendations"))
        if not recommendations:
            recommendations = self._default_recommendations(risk_level)

        summary_text = (
            str(llm_analysis.get("summary", "")).strip()
            or self._build_default_summary(company_name, summary_data, risk_level)
        )
        detailed_text = (
            str(llm_analysis.get("detailed_analysis", "")).strip()
            or self._build_default_detail(summary_data, preliminary_risk)
        )

        report = DecisionReport(
            company_name=company_name,
            stock_code=stock_code,
            analysis_period=time_range or "Last 7 days",
            overall_sentiment=summary_data["overall_sentiment"],
            risk_assessment=RiskAssessment(
                risk_level=risk_level,
                risk_score=final_score,
                risk_factors=combined_factors,
                confidence=confidence,
                reasoning=detailed_text,
            ),
            key_events=summary_data["key_events"],
            recommendations=recommendations,
            summary=summary_text,
            detailed_analysis=detailed_text,
            sentiment_distribution=summary_data["sentiment_distribution"],
            sentiment_timeline=summary_data["sentiment_timeline"],
            event_counts=summary_data["event_counts"],
            negative_event_summaries=summary_data["negative_event_summaries"],
            data_overview=summary_data["data_overview"],
        )

        self.state.status = "completed"
        self.state.output = report
        return report

    def _aggregate_results(
        self,
        analyzed_documents: List[Dict[str, Any]],
        analysis_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized_summary = self._normalize_analysis_summary(analysis_summary)
        if not analyzed_documents:
            return {
                "overall_sentiment": {"label": "neutral", "score": 0.5},
                "key_events": [],
                "sentiment_distribution": {"positive": 0, "neutral": 0, "negative": 0},
                "sentiment_timeline": [],
                "event_counts": {},
                "negative_event_summaries": [],
                "trend": "stable",
                "analysis_summary": normalized_summary,
                "data_overview": {
                    "total_documents": 0,
                    "source_distribution": {},
                    "raw_input_documents": normalized_summary.get("input_documents", 0),
                    "filtered_out_documents": normalized_summary.get("skipped_documents", 0),
                    "effective_text_limit": normalized_summary.get("effective_text_limit"),
                    "analysis_stopped_early": normalized_summary.get("analysis_stopped_early", False),
                    "processed_documents": normalized_summary.get("processed_documents", 0),
                    "unprocessed_document_count": normalized_summary.get("unprocessed_document_count", 0),
                    "text_analysis_summary": (
                        normalized_summary.get("overall_document_summary")
                        or normalized_summary.get("executive_summary", "")
                    ),
                    "overall_document_summary": (
                        normalized_summary.get("overall_document_summary")
                        or normalized_summary.get("executive_summary", "")
                    ),
                    "document_summaries": normalized_summary.get("document_summaries", []),
                },
            }

        sentiment_distribution = {"positive": 0, "neutral": 0, "negative": 0}
        event_counts: Dict[str, int] = {}
        source_distribution: Dict[str, int] = {}
        timeline_map: Dict[str, Dict[str, Any]] = {}
        all_events: List[Dict[str, Any]] = []
        signed_scores: List[float] = []

        for doc in analyzed_documents:
            original = doc.get("original", {}) if isinstance(doc, dict) else {}
            sentiment = doc.get("sentiment", {}) if isinstance(doc, dict) else {}

            label = str(sentiment.get("label", "neutral")).lower()
            score = self._safe_float(sentiment.get("score"), default=0.5)
            if label not in sentiment_distribution:
                label = "neutral"
            sentiment_distribution[label] += 1

            if label == "positive":
                signed_scores.append(score)
            elif label == "negative":
                signed_scores.append(-score)
            else:
                signed_scores.append(0.0)

            source_type = str(original.get("type") or original.get("source") or "unknown")
            source_distribution[source_type] = source_distribution.get(source_type, 0) + 1

            publish_date = self._normalize_publish_date(original.get("publish_date"))
            if publish_date not in timeline_map:
                timeline_map[publish_date] = {
                    "date": publish_date,
                    "positive": 0,
                    "neutral": 0,
                    "negative": 0,
                    "avg_signed_score": 0.0,
                    "count": 0,
                    "_sum": 0.0,
                }
            bucket = timeline_map[publish_date]
            bucket[label] += 1
            bucket["count"] += 1
            bucket["_sum"] += signed_scores[-1]

            for event in doc.get("events", []) if isinstance(doc, dict) else []:
                if not isinstance(event, dict):
                    continue
                event_type = self._normalize_event_type(event.get("type", "unknown"))
                trigger = event.get("trigger", "")
                normalized_trigger = self._normalize_event_type(trigger) if trigger else ""
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
                all_events.append(
                    {
                        "type": event_type,
                        "title": event.get("title") or f"{self._format_event_type(event_type)} related event",
                        "trigger": normalized_trigger,
                        "date": event.get("date") or publish_date,
                        "impact": event.get("impact") or event.get("arguments", {}),
                        "confidence": self._safe_float(event.get("confidence"), 0.6),
                        "source_title": original.get("title", ""),
                    }
                )

        timeline = sorted(timeline_map.values(), key=lambda x: x["date"])
        for item in timeline:
            count = item["count"] or 1
            item["avg_signed_score"] = round(item["_sum"] / count, 3)
            item.pop("_sum", None)

        avg_signed = sum(signed_scores) / len(signed_scores) if signed_scores else 0.0
        overall_score = round((avg_signed + 1.0) / 2.0, 2)
        if avg_signed > 0.12:
            overall_label = "positive"
        elif avg_signed < -0.12:
            overall_label = "negative"
        else:
            overall_label = "neutral"

        trend = self._infer_trend(timeline)
        key_events = self._select_key_events(all_events)
        negative_summaries = self._summarize_negative_events(key_events)

        return {
            "overall_sentiment": {"label": overall_label, "score": overall_score},
            "key_events": key_events,
            "sentiment_distribution": sentiment_distribution,
            "sentiment_timeline": timeline,
            "event_counts": event_counts,
            "negative_event_summaries": negative_summaries,
            "trend": trend,
            "analysis_summary": normalized_summary,
            "data_overview": {
                "total_documents": len(analyzed_documents),
                "source_distribution": source_distribution,
                "key_event_count": len(key_events),
                "raw_input_documents": normalized_summary.get("input_documents", len(analyzed_documents)),
                "filtered_out_documents": normalized_summary.get("skipped_documents", 0),
                "effective_text_limit": normalized_summary.get("effective_text_limit"),
                "analysis_stopped_early": normalized_summary.get("analysis_stopped_early", False),
                "processed_documents": normalized_summary.get("processed_documents", len(analyzed_documents)),
                "unprocessed_document_count": normalized_summary.get("unprocessed_document_count", 0),
                "text_analysis_summary": (
                    normalized_summary.get("overall_document_summary")
                    or normalized_summary.get("executive_summary", "")
                ),
                "overall_document_summary": (
                    normalized_summary.get("overall_document_summary")
                    or normalized_summary.get("executive_summary", "")
                ),
                "document_summaries": normalized_summary.get("document_summaries", []),
            },
        }

    def _normalize_analysis_summary(self, analysis_summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(analysis_summary, dict):
            return {
                "input_documents": 0,
                "kept_documents": 0,
                "skipped_documents": 0,
                "effective_text_limit": None,
                "analysis_stopped_early": False,
                "processed_documents": 0,
                "unprocessed_document_count": 0,
                "document_summaries": [],
                "overall_document_summary": "",
                "executive_summary": "",
            }

        return {
            "target_company": analysis_summary.get("target_company"),
            "input_documents": int(self._safe_float(analysis_summary.get("input_documents"), 0) or 0),
            "kept_documents": int(self._safe_float(analysis_summary.get("kept_documents"), 0) or 0),
            "skipped_documents": int(self._safe_float(analysis_summary.get("skipped_documents"), 0) or 0),
            "effective_text_limit": (
                int(self._safe_float(analysis_summary.get("effective_text_limit"), 0) or 0)
                if analysis_summary.get("effective_text_limit") is not None
                else None
            ),
            "analysis_stopped_early": bool(analysis_summary.get("analysis_stopped_early", False)),
            "processed_documents": int(self._safe_float(analysis_summary.get("processed_documents"), 0) or 0),
            "unprocessed_document_count": int(
                self._safe_float(analysis_summary.get("unprocessed_document_count"), 0) or 0
            ),
            "sentiment_distribution": analysis_summary.get("sentiment_distribution", {}),
            "top_event_types": analysis_summary.get("top_event_types", []),
            "top_keywords": analysis_summary.get("top_keywords", []),
            "highlights": analysis_summary.get("highlights", []),
            "document_summaries": self._normalize_document_summaries(analysis_summary.get("document_summaries")),
            "overall_document_summary": str(analysis_summary.get("overall_document_summary", "")).strip(),
            "executive_summary": str(analysis_summary.get("executive_summary", "")).strip(),
        }

    def _normalize_document_summaries(self, value: Any) -> List[Dict[str, Any]]:
        if not isinstance(value, list):
            return []

        normalized: List[Dict[str, Any]] = []
        for idx, item in enumerate(value, start=1):
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "index": int(self._safe_float(item.get("index"), idx) or idx),
                    "title": str(item.get("title", "")).strip(),
                    "summary": str(item.get("summary", "")).strip(),
                    "source": str(item.get("source", "")).strip(),
                    "publish_date": str(item.get("publish_date", "")).strip(),
                    "url": str(item.get("url", "")).strip(),
                    "sentiment_label": str(item.get("sentiment_label", "neutral")).strip().lower() or "neutral",
                    "sentiment_score": round(float(self._safe_float(item.get("sentiment_score"), 0.5) or 0.5), 3),
                }
            )
        return normalized

    def _normalize_publish_date(self, value: Any) -> str:
        if not value:
            return datetime.now().strftime("%Y-%m-%d")
        text = str(value).strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            return text
        candidate = re.search(r"\d{4}-\d{1,2}-\d{1,2}", text)
        if candidate:
            y, m, d = candidate.group().split("-")
            return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return parsed.strftime("%Y-%m-%d")
        except Exception:
            return datetime.now().strftime("%Y-%m-%d")

    def _infer_trend(self, timeline: List[Dict[str, Any]]) -> str:
        if len(timeline) <= 1:
            return "stable"
        mid = len(timeline) // 2
        first = timeline[:mid] if mid else timeline[:1]
        second = timeline[mid:] if mid else timeline[1:]

        def avg_negative_ratio(items: List[Dict[str, Any]]) -> float:
            total = sum(it.get("count", 0) for it in items) or 1
            neg = sum(it.get("negative", 0) for it in items)
            return neg / total

        first_neg = avg_negative_ratio(first)
        second_neg = avg_negative_ratio(second or first)
        if second_neg - first_neg > 0.12:
            return "deteriorating"
        if first_neg - second_neg > 0.12:
            return "improving"
        return "stable"

    def _select_key_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not events:
            return []

        def score(event: Dict[str, Any]) -> float:
            e_type = self._normalize_event_type(event.get("type", ""))
            confidence = self._safe_float(event.get("confidence"), 0.6)
            base = EVENT_TYPE_RISK_WEIGHTS.get(e_type, 0.4)
            return base * 0.65 + confidence * 0.35

        ranked = sorted(events, key=score, reverse=True)
        unique: List[Dict[str, Any]] = []
        seen = set()
        for event in ranked:
            key = (event.get("type"), event.get("trigger"), event.get("date"))
            if key in seen:
                continue
            seen.add(key)
            unique.append(event)
            if len(unique) >= 12:
                break
        return unique

    def _summarize_negative_events(self, key_events: List[Dict[str, Any]]) -> List[str]:
        summaries = []
        for event in key_events:
            if self._normalize_event_type(event.get("type")) not in HIGH_RISK_EVENT_TYPES:
                continue
            date = event.get("date", "")
            title = event.get("title") or event.get("trigger") or event.get("type")
            impact = str(event.get("impact", ""))
            snippet = impact.strip()
            if len(snippet) > 200:
                snippet = f"{snippet[:200]}..."
            summaries.append(f"{date} {title}: {snippet}")
        return summaries[:5]

    async def _rule_based_assessment(
        self,
        summary_data: Dict[str, Any],
        memory_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        score = 0.35
        factors: List[str] = []

        sentiment_dist = summary_data.get("sentiment_distribution", {})
        total = sum(sentiment_dist.values()) or 1
        negative_ratio = sentiment_dist.get("negative", 0) / total

        if negative_ratio >= 0.5:
            score += 0.22
            factors.append(f"High negative sentiment ratio ({negative_ratio:.0%})")
        elif negative_ratio >= 0.3:
            score += 0.12
            factors.append(f"Elevated negative sentiment ratio ({negative_ratio:.0%})")

        event_counts = summary_data.get("event_counts", {})
        high_event_count = sum(
            count
            for event_type, count in event_counts.items()
            if self._normalize_event_type(event_type) in HIGH_RISK_EVENT_TYPES
        )
        if high_event_count > 0:
            score += min(0.25, 0.08 + high_event_count * 0.05)
            factors.append(f"High-risk event occurrences: {high_event_count}")

        total_events = sum(event_counts.values())
        if total_events >= 10:
            score += 0.1
            factors.append("Event volume remains elevated")

        trend = summary_data.get("trend", "stable")
        if trend == "deteriorating":
            score += 0.14
            factors.append("Sentiment trend continues to deteriorate")

        history_penalty = self._history_penalty(memory_context, negative_ratio)
        if history_penalty > 0:
            score += history_penalty
            factors.append("Higher risk versus recent historical baseline")

        score = round(min(1.0, max(0.0, score)), 2)
        return {
            "risk_level": self._score_to_level(score),
            "risk_score": score,
            "risk_factors": factors,
            "method": "rule_based",
        }

    def _history_penalty(self, memory_context: Dict[str, Any], current_negative_ratio: float) -> float:
        reports = memory_context.get("recent_reports", []) if isinstance(memory_context, dict) else []
        if not reports:
            return 0.0

        historical_ratios = []
        for item in reports[:6]:
            dist = item.get("sentiment_distribution", {})
            total = sum(dist.values()) or 1
            historical_ratios.append(dist.get("negative", 0) / total)
        if not historical_ratios:
            return 0.0

        avg_hist = sum(historical_ratios) / len(historical_ratios)
        delta = current_negative_ratio - avg_hist
        if delta > 0.2:
            return 0.12
        if delta > 0.1:
            return 0.06
        return 0.0

    async def _llm_analysis(
        self,
        company_name: str,
        stock_code: Optional[str],
        summary_data: Dict[str, Any],
        preliminary_risk: Dict[str, Any],
        memory_context: Dict[str, Any],
        documents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        try:
            prompt = self._build_analysis_prompt(
                company_name=company_name,
                stock_code=stock_code,
                summary_data=summary_data,
                preliminary_risk=preliminary_risk,
                memory_context=memory_context,
                documents=documents,
            )
            response = await self.llm.ainvoke(prompt)
            content = getattr(response, "content", "") or ""
            parsed = self._extract_json(content)
            return parsed or {}
        except Exception as exc:
            logger.warning("LLM refinement skipped: %s", exc)
            return {}

    def _build_analysis_prompt(
        self,
        company_name: str,
        stock_code: Optional[str],
        summary_data: Dict[str, Any],
        preliminary_risk: Dict[str, Any],
        memory_context: Dict[str, Any],
        documents: List[Dict[str, Any]],
    ) -> str:
        doc_lines = []
        for idx, doc in enumerate(documents[:5], start=1):
            original = doc.get("original", {})
            sentiment = doc.get("sentiment", {})
            events = [e.get("type") for e in doc.get("events", []) if isinstance(e, dict)]
            doc_lines.append(
                f"{idx}. Title: {original.get('title', '')}\n"
                f"   Sentiment: {sentiment.get('label', 'neutral')} ({sentiment.get('score', 0.5)})\n"
                f"   Events: {events}\n"
                f"   Summary: {doc.get('summary', '')[:120]}"
            )

        memory_lines = []
        for item in (memory_context.get("recent_reports", []) or [])[:3]:
            risk = item.get("risk_assessment", {})
            memory_lines.append(
                f"- {item.get('generated_at', '')[:10]} risk={risk.get('risk_level', 'unknown')} score={risk.get('risk_score', 'n/a')}"
            )

        analysis_summary = summary_data.get("analysis_summary", {}) if isinstance(summary_data, dict) else {}
        analysis_summary_text = str(analysis_summary.get("executive_summary", "")).strip() or "none"
        analysis_filter_text = (
            f"{analysis_summary.get('skipped_documents', 0)} / {analysis_summary.get('input_documents', 0)}"
            if isinstance(analysis_summary, dict)
            else "0/0"
        )

        return (
            "You are a listed-company sentiment risk analyst. "
            "Based on the provided data, output final judgment in JSON only.\n"
            f"Company: {company_name}\n"
            f"Stock Code: {stock_code or 'unknown'}\n"
            f"Document Count: {summary_data.get('data_overview', {}).get('total_documents', 0)}\n"
            f"Sentiment Distribution: {summary_data.get('sentiment_distribution', {})}\n"
            f"Sentiment Trend: {summary_data.get('trend', 'stable')}\n"
            f"Event Counts: {summary_data.get('event_counts', {})}\n"
            f"Text-analysis Summary: {analysis_summary_text}\n"
            f"Filtered Docs (text-analysis stage): {analysis_filter_text}\n"
            f"Rule-based Preliminary Result: {preliminary_risk}\n"
            f"Historical Records: {memory_lines or 'none'}\n\n"
            f"Sample Documents:\n{chr(10).join(doc_lines) if doc_lines else 'none'}\n\n"
            "Return JSON with this schema:\n"
            "{\n"
            '  "risk_level": "low/medium/high/critical",\n'
            '  "risk_score": 0.0,\n'
            '  "risk_factors": ["factor1", "factor2"],\n'
            '  "summary": "short summary",\n'
            '  "detailed_analysis": "detailed analysis",\n'
            '  "recommendations": ["rec1", "rec2"]\n'
            "}\n"
        )

    def _extract_json(self, text: str) -> Dict[str, Any]:
        if not text:
            return {}
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {}
        candidate = match.group(0)
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def _score_to_level(self, score: float) -> str:
        if score >= 0.8:
            return "critical"
        if score >= 0.62:
            return "high"
        if score >= 0.42:
            return "medium"
        return "low"

    def _normalize_event_type(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return "unknown"
        return EVENT_TYPE_LOOKUP.get(text.lower(), text.lower())

    def _format_event_type(self, value: Any) -> str:
        normalized = self._normalize_event_type(value)
        return EVENT_TYPE_DISPLAY.get(normalized, normalized.replace("_", " ").title())

    def _safe_float(self, value: Any, default: Optional[float] = 0.0) -> Optional[float]:
        if value is None:
            return default
        try:
            return float(value)
        except Exception:
            return default

    def _safe_str_list(self, value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    def _merge_text_lists(self, first: List[str], second: List[str]) -> List[str]:
        merged = []
        seen = set()
        for item in [*(first or []), *(second or [])]:
            text = str(item).strip()
            if not text:
                continue
            if text in seen:
                continue
            seen.add(text)
            merged.append(text)
        return merged[:8]

    def _build_default_summary(
        self,
        company_name: str,
        summary_data: Dict[str, Any],
        risk_level: str,
    ) -> str:
        dist = summary_data.get("sentiment_distribution", {})
        neg = dist.get("negative", 0)
        total = sum(dist.values()) or 1
        neg_ratio = neg / total
        trend = summary_data.get("trend", "stable")
        analysis_text = str(summary_data.get("analysis_summary", {}).get("executive_summary", "")).strip()
        base = (
            f"{company_name} current sentiment risk level is {risk_level}. "
            f"Negative sentiment ratio is approximately {neg_ratio:.0%}, with trend {trend}. "
            "Recommend continuous monitoring of announcements and high-risk events."
        )
        if analysis_text:
            return f"{base} Text-analysis summary: {analysis_text}"
        return base

    def _build_default_detail(
        self,
        summary_data: Dict[str, Any],
        preliminary_risk: Dict[str, Any],
    ) -> str:
        factors = preliminary_risk.get("risk_factors", [])
        factor_text = "; ".join([str(item).strip() for item in factors if str(item).strip()]) or "No major risk factor."
        detail = (
            f"Rule-based assessment result: level={preliminary_risk.get('risk_level')}, "
            f"score={preliminary_risk.get('risk_score')}. "
            f"Main reasons: {factor_text}."
        )
        analysis_text = str(summary_data.get("analysis_summary", {}).get("executive_summary", "")).strip()
        if analysis_text:
            detail = f"{detail} Text-analysis summary: {analysis_text}"
        return detail

    def _default_recommendations(self, risk_level: str) -> List[str]:
        if risk_level in {"high", "critical"}:
            return [
                "Track newly emerged negative information daily and use alerts.",
                "Verify facts and progress of penalties, lawsuits, and investigations.",
                "Prepare investor-relations responses to reduce sentiment spillover.",
            ]
        if risk_level == "medium":
            return [
                "Continue tracking announcements and media sentiment changes.",
                "Watch whether key events trigger secondary dissemination.",
            ]
        return [
            "Current risk appears controllable; keep routine monitoring cadence.",
            "Regularly review keyword and sentiment distribution for drift.",
        ]


def create_decision_agent(
    llm: BaseLanguageModel,
    tools: Optional[List[BaseTool]] = None,
) -> Any:
    """
    Create a simplified ReAct decision agent.
    """
    from langchain import hub

    try:
        from langchain.agents import create_react_agent
        try:
            from langchain.agents import AgentExecutor
        except ImportError:
            from langchain.agents.agent import AgentExecutor
    except ImportError as exc:  # pragma: no cover - optional path
        raise RuntimeError("Current langchain version does not support ReAct helper APIs.") from exc

    all_tools = tools or []
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=all_tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=all_tools,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True,
    )
