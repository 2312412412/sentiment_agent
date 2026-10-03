"""
Text analysis agent.

Responsibilities:
- entity recognition
- event extraction
- sentiment classification
- summary and keyword extraction
"""
import asyncio
import hashlib
import logging
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

from .base_agent import BaseAgent

try:
    from ..tools import entity_recognition, event_extraction, sentiment_analysis
except ImportError:  # pragma: no cover - script mode fallback
    from tools import entity_recognition, event_extraction, sentiment_analysis

logger = logging.getLogger(__name__)


def _unique_tools(tools: List[BaseTool]) -> List[BaseTool]:
    unique: List[BaseTool] = []
    seen = set()
    for tool in tools:
        name = getattr(tool, "name", id(tool))
        if name in seen:
            continue
        seen.add(name)
        unique.append(tool)
    return unique


class TextAnalysisAgent(BaseAgent):
    """
    Analyze collected documents and produce structured outputs.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[Any] = None,
        config: Optional[Any] = None,
    ):
        default_tools = [sentiment_analysis, entity_recognition, event_extraction]
        super().__init__(
            name="text_analyst",
            llm=llm,
            tools=_unique_tools(default_tools + (tools or [])),
            memory=memory,
            config=config,
        )
        self._analysis_cache: Dict[str, Dict[str, Any]] = {}
        self._last_analysis_summary: Dict[str, Any] = {}

    async def execute(
        self,
        documents: List[Dict[str, Any]],
        target_company: Optional[str] = None,
        batch_size: int = 8,
        max_effective_documents: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        self.state.task = f"analyze {len(documents)} documents"
        self.state.status = "running"
        effective_limit = self._normalize_effective_limit(max_effective_documents)

        if not documents:
            summary = self._empty_analysis_summary(
                target_company=target_company,
                input_documents=0,
                effective_text_limit=effective_limit,
            )
            self._last_analysis_summary = summary
            self.state.context["analysis_summary"] = summary
            self.state.context["skipped_documents"] = []
            self.state.status = "completed"
            self.state.output = []
            return []

        analyzed_docs, skipped_docs = await self._analyze_batch(
            documents=documents,
            target_company=target_company,
            batch_size=max(1, batch_size),
            max_effective_documents=effective_limit,
        )
        summary = await self._build_analysis_summary(
            analyzed_documents=analyzed_docs,
            skipped_documents=skipped_docs,
            target_company=target_company,
            input_documents=len(documents),
            effective_text_limit=effective_limit,
        )
        self._last_analysis_summary = summary
        self.state.context["analysis_summary"] = summary
        self.state.context["skipped_documents"] = skipped_docs
        self.state.status = "completed"
        self.state.output = analyzed_docs
        return analyzed_docs

    async def _analyze_batch(
        self,
        documents: List[Dict[str, Any]],
        target_company: Optional[str],
        batch_size: int,
        max_effective_documents: Optional[int],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        results: List[Dict[str, Any]] = []
        skipped_docs: List[Dict[str, Any]] = []
        if max_effective_documents is not None:
            for idx, doc in enumerate(documents):
                if len(results) >= max_effective_documents:
                    break

                try:
                    result = await self._analyze_single(doc, target_company)
                except Exception as exc:
                    logger.warning("document analysis failed: %s", exc)
                    result = {
                        "original": doc,
                        "sentiment": {"label": "neutral", "score": 0.5, "aspects": {}},
                        "entities": [],
                        "events": [],
                        "summary": "",
                        "keywords": [],
                        "is_relevant": True,
                        "error": str(exc),
                        "analyzed_at": datetime.now().isoformat(),
                    }

                if result.get("is_relevant", True):
                    results.append(result)
                else:
                    skipped_docs.append(
                        self._build_skipped_record(
                            doc=doc,
                            reason=str(result.get("skip_reason", "not related to target company")),
                            matched_terms=result.get("relevance_matches", []),
                        )
                    )

                if len(results) >= max_effective_documents:
                    break

                if idx + 1 < len(documents):
                    await asyncio.sleep(0.05)
            return results, skipped_docs

        for idx in range(0, len(documents), batch_size):
            batch = documents[idx : idx + batch_size]
            tasks = [self._analyze_single(doc, target_company) for doc in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for doc, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.warning("document analysis failed: %s", result)
                    results.append(
                        {
                            "original": doc,
                            "sentiment": {"label": "neutral", "score": 0.5, "aspects": {}},
                            "entities": [],
                            "events": [],
                            "summary": "",
                            "keywords": [],
                            "is_relevant": True,
                            "error": str(result),
                            "analyzed_at": datetime.now().isoformat(),
                        }
                    )
                else:
                    if result.get("is_relevant", True):
                        results.append(result)
                    else:
                        skipped_docs.append(
                            self._build_skipped_record(
                                doc=doc,
                                reason=str(result.get("skip_reason", "not related to target company")),
                                matched_terms=result.get("relevance_matches", []),
                            )
                        )

            if idx + batch_size < len(documents):
                await asyncio.sleep(0.2)
        return results, skipped_docs

    async def _analyze_single(
        self,
        doc: Dict[str, Any],
        target_company: Optional[str],
    ) -> Dict[str, Any]:
        content = str(doc.get("content", "") or "").strip()
        title = str(doc.get("title", "") or "").strip()
        unreadable_reason = self._detect_unreadable_reason(doc=doc, content=content)
        text_blob = self._build_doc_text_blob(title=title, content=content)
        target_terms = self._prepare_target_terms(doc, target_company)
        cache_key = self._build_cache_key(doc, target_company)
        if cache_key in self._analysis_cache:
            cached = dict(self._analysis_cache[cache_key])
            cached["original"] = doc
            return cached

        if unreadable_reason:
            skipped_output = {
                "original": doc,
                "sentiment": {"label": "neutral", "score": 0.5, "aspects": {}},
                "entities": [],
                "events": [],
                "summary": "",
                "keywords": [],
                "is_relevant": False,
                "skip_reason": unreadable_reason,
                "relevance_matches": [],
                "analyzed_at": datetime.now().isoformat(),
            }
            self._analysis_cache[cache_key] = skipped_output
            return skipped_output

        entities = []
        if content:
            entities = await self._recognize_entities(content, target_company, target_terms)

        is_relevant, relevance_detail = self._is_related_to_target(
            doc=doc,
            text_blob=text_blob,
            entities=entities,
            target_terms=target_terms,
        )
        if not is_relevant:
            skipped_output = {
                "original": doc,
                "sentiment": {"label": "neutral", "score": 0.5, "aspects": {}},
                "entities": entities,
                "events": [],
                "summary": "",
                "keywords": [],
                "is_relevant": False,
                "skip_reason": relevance_detail.get("reason", "not related to target company"),
                "relevance_matches": relevance_detail.get("matched_terms", []),
                "analyzed_at": datetime.now().isoformat(),
            }
            self._analysis_cache[cache_key] = skipped_output
            return skipped_output

        if not content:
            output = {
                "original": doc,
                "sentiment": {"label": "neutral", "score": 0.5, "aspects": {}},
                "entities": entities,
                "events": [],
                "summary": "",
                "keywords": [],
                "is_relevant": True,
                "relevance_matches": relevance_detail.get("matched_terms", []),
                "analyzed_at": datetime.now().isoformat(),
            }
            self._analysis_cache[cache_key] = output
            return output

        events = await self._extract_events(content, entities)
        sentiment = await self._analyze_sentiment(content)
        summary = await self._summarize(content)
        keywords = await self._extract_keywords(content)

        analyzed = {
            "original": doc,
            "sentiment": sentiment,
            "entities": entities,
            "events": events,
            "summary": summary,
            "keywords": keywords,
            "is_relevant": True,
            "relevance_matches": relevance_detail.get("matched_terms", []),
            "analyzed_at": datetime.now().isoformat(),
        }
        self._analysis_cache[cache_key] = analyzed
        return analyzed

    def get_analysis_summary(self) -> Dict[str, Any]:
        return dict(self._last_analysis_summary)

    def _empty_analysis_summary(
        self,
        target_company: Optional[str],
        input_documents: int,
        effective_text_limit: Optional[int],
    ) -> Dict[str, Any]:
        target = (target_company or "").strip() or "target company"
        return {
            "target_company": target_company,
            "input_documents": input_documents,
            "kept_documents": 0,
            "skipped_documents": 0,
            "effective_text_limit": effective_text_limit,
            "analysis_stopped_early": False,
            "processed_documents": 0,
            "unprocessed_document_count": 0,
            "skipped_titles": [],
            "sentiment_distribution": {"positive": 0, "neutral": 0, "negative": 0},
            "top_event_types": [],
            "top_keywords": [],
            "highlights": [],
            "document_summaries": [],
            "overall_document_summary": f"No valid text was analyzed for {target}.",
            "executive_summary": f"No valid text was analyzed for {target}.",
            "generated_at": datetime.now().isoformat(),
        }

    def _build_doc_text_blob(self, title: str, content: str) -> str:
        return " ".join([part.strip() for part in [title, content] if part and part.strip()])

    def _normalize_company_term(self, term: str) -> str:
        text = str(term or "").strip().lower()
        if not text:
            return ""
        text = re.sub(r"[\s\-_.,，。:：;；!?！？\"'“”‘’`~（）()【】\[\]<>《》/\\]+", "", text)
        return text

    def _detect_unreadable_reason(self, doc: Dict[str, Any], content: str) -> str:
        metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
        if isinstance(metadata, dict) and metadata.get("unreadable_content"):
            return str(metadata.get("unreadable_reason") or "unreadable_content").strip()

        sample = str(content or "")[:12000]
        if not sample:
            return ""

        lowered = sample.lower()
        pdf_markers = ["%pdf", "endobj", "stream", "endstream", "xref", "trailer", "/length", "/filter"]
        marker_hits = sum(1 for marker in pdf_markers if marker in lowered)
        if ("%pdf" in lowered and marker_hits >= 3) or marker_hits >= 6:
            return "pdf_structure_stream"

        if re.search(r"(?:\\x[0-9a-fA-F]{2}){8,}", sample):
            return "hex_encoded_binary_stream"

        if len(sample) >= 300:
            allowed_chars = sum(
                1
                for ch in sample
                if (
                    ch.isalnum()
                    or "\u4e00" <= ch <= "\u9fff"
                    or ch in " \t\r\n,.;:!?，。；：、（）()【】[]《》<>\"'“”‘’-_/"
                )
            )
            symbol_chars = sum(1 for ch in sample if ch in "{}<>/\\|@#$%^&*_+=~`")
            allowed_ratio = allowed_chars / max(1, len(sample))
            symbol_ratio = symbol_chars / max(1, len(sample))
            if allowed_ratio < 0.55 and symbol_ratio > 0.14:
                return "binary_or_encoded_payload"

        return ""

    def _prepare_target_terms(
        self,
        doc: Dict[str, Any],
        target_company: Optional[str],
    ) -> List[str]:
        metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
        aliases = metadata.get("aliases", []) if isinstance(metadata, dict) else []

        raw_terms: List[str] = [
            str(target_company or "").strip(),
            str(doc.get("company_name", "") or "").strip(),
            str(doc.get("stock_code", "") or "").strip(),
        ]
        raw_terms.extend([str(item).strip() for item in aliases if str(item).strip()])

        expanded_terms: List[str] = []
        suffixes = ["股份有限公司", "有限责任公司", "有限公司", "集团股份", "集团", "控股", "股份"]
        for term in raw_terms:
            if not term:
                continue
            expanded_terms.append(term)
            for suffix in suffixes:
                if term.endswith(suffix):
                    trimmed = term[: -len(suffix)].strip()
                    if trimmed:
                        expanded_terms.append(trimmed)

        cleaned: List[str] = []
        seen = set()
        generic_tokens = {"公司", "企业", "集团"}
        for term in expanded_terms:
            normalized = self._normalize_company_term(term)
            if not normalized or normalized in seen:
                continue
            if normalized in generic_tokens:
                continue
            if not normalized.isdigit() and len(normalized) < 2:
                continue
            seen.add(normalized)
            cleaned.append(term.strip())
        return cleaned

    def _is_related_to_target(
        self,
        doc: Dict[str, Any],
        text_blob: str,
        entities: List[Dict[str, Any]],
        target_terms: List[str],
    ) -> Tuple[bool, Dict[str, Any]]:
        if not target_terms:
            return True, {"reason": "no_target_company", "matched_terms": []}

        normalized_blob = self._normalize_company_term(text_blob)
        matched_terms: List[str] = []
        seen_matches = set()
        for term in target_terms:
            normalized_term = self._normalize_company_term(term)
            if not normalized_term:
                continue
            if normalized_term.isdigit():
                hit = normalized_term in text_blob
            else:
                hit = normalized_term in normalized_blob
            if hit and normalized_term not in seen_matches:
                seen_matches.add(normalized_term)
                matched_terms.append(term)

        for entity in entities:
            entity_text = str(entity.get("text", "")).strip()
            if not entity_text:
                continue
            normalized_entity = self._normalize_company_term(entity_text)
            if not normalized_entity:
                continue
            if any(
                normalized_entity in self._normalize_company_term(term)
                or self._normalize_company_term(term) in normalized_entity
                for term in target_terms
            ) and normalized_entity not in seen_matches:
                seen_matches.add(normalized_entity)
                matched_terms.append(entity_text)

        if matched_terms:
            return True, {"reason": "matched_target_terms", "matched_terms": matched_terms}

        collector_score = 0.0
        try:
            collector_score = float(doc.get("relevance_score", 0.0))
        except Exception:
            collector_score = 0.0
        if collector_score >= 0.85:
            return True, {
                "reason": f"high_collection_relevance_score({collector_score:.2f})",
                "matched_terms": [],
            }

        return False, {"reason": "missing_target_entity_or_keyword", "matched_terms": []}

    def _build_skipped_record(
        self,
        doc: Dict[str, Any],
        reason: str,
        matched_terms: List[str],
    ) -> Dict[str, Any]:
        return {
            "title": str(doc.get("title", "") or "").strip(),
            "url": str(doc.get("url", "") or "").strip(),
            "source": str(doc.get("source", "") or doc.get("type", "")).strip(),
            "publish_date": str(doc.get("publish_date", "") or "").strip(),
            "reason": reason,
            "matched_terms": matched_terms,
        }

    async def _build_analysis_summary(
        self,
        analyzed_documents: List[Dict[str, Any]],
        skipped_documents: List[Dict[str, Any]],
        target_company: Optional[str],
        input_documents: int,
        effective_text_limit: Optional[int],
    ) -> Dict[str, Any]:
        processed_documents = len(analyzed_documents) + len(skipped_documents)
        unprocessed_document_count = max(0, input_documents - processed_documents)
        analysis_stopped_early = unprocessed_document_count > 0

        if not analyzed_documents:
            empty = self._empty_analysis_summary(
                target_company=target_company,
                input_documents=input_documents,
                effective_text_limit=effective_text_limit,
            )
            empty["skipped_documents"] = len(skipped_documents)
            empty["processed_documents"] = processed_documents
            empty["unprocessed_document_count"] = unprocessed_document_count
            empty["analysis_stopped_early"] = analysis_stopped_early
            empty["skipped_titles"] = [item.get("title", "") for item in skipped_documents[:5]]
            target = (target_company or "").strip() or "target company"
            empty["overall_document_summary"] = (
                f"Received {input_documents} documents. "
                f"All of them were skipped because they are not strongly related to {target}."
            )
            empty["executive_summary"] = empty["overall_document_summary"]
            return empty

        sentiment_distribution = {"positive": 0, "neutral": 0, "negative": 0}
        event_counter: Counter[str] = Counter()
        keyword_counter: Counter[str] = Counter()
        document_summaries = self._build_document_summaries(analyzed_documents)
        highlights = [
            f"{item.get('title', '')}: {item.get('summary', '')}".strip(": ")
            for item in document_summaries
            if item.get("title") or item.get("summary")
        ]

        for doc in analyzed_documents:
            sentiment = doc.get("sentiment", {}) if isinstance(doc, dict) else {}
            label = str(sentiment.get("label", "neutral")).lower()
            if label not in sentiment_distribution:
                label = "neutral"
            sentiment_distribution[label] += 1

            for event in doc.get("events", []) if isinstance(doc, dict) else []:
                if not isinstance(event, dict):
                    continue
                event_type = str(event.get("type", "unknown")).strip() or "unknown"
                event_counter[event_type] += 1

            for keyword in doc.get("keywords", []) if isinstance(doc, dict) else []:
                clean_keyword = str(keyword).strip()
                if clean_keyword:
                    keyword_counter[clean_keyword] += 1

        top_event_types = [
            {"type": event_type, "count": count}
            for event_type, count in event_counter.most_common(5)
        ]
        top_keywords = [word for word, _ in keyword_counter.most_common(8)]
        skipped_count = len(skipped_documents)
        kept_count = len(analyzed_documents)
        overall_document_summary = await self._generate_overall_document_summary(
            document_summaries=document_summaries,
            target_company=target_company,
        )

        return {
            "target_company": target_company,
            "input_documents": input_documents,
            "kept_documents": kept_count,
            "skipped_documents": skipped_count,
            "effective_text_limit": effective_text_limit,
            "analysis_stopped_early": analysis_stopped_early,
            "processed_documents": processed_documents,
            "unprocessed_document_count": unprocessed_document_count,
            "skipped_titles": [item.get("title", "") for item in skipped_documents[:5]],
            "sentiment_distribution": sentiment_distribution,
            "top_event_types": top_event_types,
            "top_keywords": top_keywords,
            "highlights": highlights[:5],
            "document_summaries": document_summaries,
            "overall_document_summary": overall_document_summary,
            "executive_summary": overall_document_summary,
            "generated_at": datetime.now().isoformat(),
        }

    def _build_document_summaries(self, analyzed_documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        for idx, doc in enumerate(analyzed_documents, start=1):
            if not isinstance(doc, dict):
                continue

            original = doc.get("original", {})
            if not isinstance(original, dict):
                original = {}

            summary = str(doc.get("summary", "")).strip()
            if not summary:
                summary = str(original.get("content", "")).strip()[:120]

            title = str(original.get("title", "")).strip() or f"Document {idx}"
            sentiment = doc.get("sentiment", {})
            if not isinstance(sentiment, dict):
                sentiment = {}
            try:
                sentiment_score = round(float(sentiment.get("score", 0.5)), 3)
            except (TypeError, ValueError):
                sentiment_score = 0.5

            summaries.append(
                {
                    "index": idx,
                    "title": title,
                    "summary": summary,
                    "source": str(original.get("source", "") or original.get("type", "")).strip(),
                    "publish_date": str(original.get("publish_date", "")).strip(),
                    "url": str(original.get("url", "")).strip(),
                    "sentiment_label": str(sentiment.get("label", "neutral")).strip().lower() or "neutral",
                    "sentiment_score": sentiment_score,
                }
            )
        return summaries

    async def _generate_overall_document_summary(
        self,
        document_summaries: List[Dict[str, Any]],
        target_company: Optional[str],
    ) -> str:
        if not document_summaries:
            target = (target_company or "").strip() or "target company"
            return f"No related document summaries are available for {target}."

        summary_lines = []
        total_chars = 0
        for item in document_summaries:
            title = str(item.get("title", "")).strip() or "Untitled"
            summary = str(item.get("summary", "")).strip() or title
            sentiment_label = str(item.get("sentiment_label", "neutral")).strip() or "neutral"
            line = f"{item.get('index', len(summary_lines) + 1)}. [{sentiment_label}] {title}: {summary}"
            if summary_lines and total_chars + len(line) > 6000:
                break
            summary_lines.append(line)
            total_chars += len(line)

        try:
            prompt = ChatPromptTemplate.from_template(
                "You are a text analysis assistant.\n"
                "Write one concise overall summary based only on the document summaries below.\n"
                "Requirements:\n"
                "1. Focus on recurring themes and the main positive or negative signals.\n"
                "2. Do not invent facts.\n"
                "3. Use at most two sentences.\n"
                "4. Use the same language as the document summaries.\n\n"
                "Target company: {target_company}\n"
                "Document summaries:\n{document_summaries}\n"
            )
            chain = prompt | self.llm
            response = await chain.ainvoke(
                {
                    "target_company": (target_company or "").strip() or "target company",
                    "document_summaries": "\n".join(summary_lines),
                }
            )
            text = self._response_to_text(response).strip()
            if text:
                return text
        except Exception as exc:
            logger.warning("overall document summary failed: %s", exc)

        return self._fallback_overall_document_summary(document_summaries, target_company)

    def _fallback_overall_document_summary(
        self,
        document_summaries: List[Dict[str, Any]],
        target_company: Optional[str],
    ) -> str:
        target = (target_company or "").strip() or "target company"
        sentiment_counter = Counter(
            str(item.get("sentiment_label", "neutral")).strip().lower() or "neutral"
            for item in document_summaries
        )
        dominant_sentiment = sentiment_counter.most_common(1)[0][0] if sentiment_counter else "neutral"

        snippets: List[str] = []
        seen = set()
        for item in document_summaries:
            snippet = str(item.get("summary", "")).strip()
            if not snippet or snippet in seen:
                continue
            seen.add(snippet)
            snippets.append(snippet)
            if len(snippets) >= 3:
                break

        if not snippets:
            return (
                f"Analyzed {len(document_summaries)} related documents for {target}. "
                f"Overall sentiment leaned {dominant_sentiment}."
            )

        return (
            f"Across {len(document_summaries)} related documents for {target}, the main points were: "
            f"{'; '.join(snippets)}. "
            f"Overall sentiment leaned {dominant_sentiment}."
        )

    def _normalize_effective_limit(self, value: Optional[Any]) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return None
        return normalized if normalized >= 1 else None

    def _build_cache_key(self, doc: Dict[str, Any], target_company: Optional[str]) -> str:
        raw = "|".join(
            [
                str(doc.get("title", "")),
                str(doc.get("content", "")),
                str(doc.get("publish_date", "")),
                target_company or "",
            ]
        )
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    async def _recognize_entities(
        self,
        content: str,
        target_company: Optional[str],
        target_terms: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        try:
            entities = await entity_recognition.ainvoke({"text": content})
            entities = entities or []
            terms = target_terms or ([target_company] if target_company else [])
            if terms:
                for entity in entities:
                    text = str(entity.get("text", "")).strip()
                    if not text:
                        continue
                    normalized_text = self._normalize_company_term(text)
                    if any(
                        normalized_text
                        and (
                            normalized_text in self._normalize_company_term(term)
                            or self._normalize_company_term(term) in normalized_text
                        )
                        for term in terms
                    ):
                        entity["is_target"] = True
            return entities
        except Exception as exc:
            logger.warning("entity recognition failed: %s", exc)
            return []

    async def _extract_events(
        self,
        content: str,
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        try:
            entity_texts = [str(item.get("text", "")).strip() for item in entities if item.get("text")]
            events = await event_extraction.ainvoke({"text": content, "entities": entity_texts})
            return events or []
        except Exception as exc:
            logger.warning("event extraction failed: %s", exc)
            return []

    async def _analyze_sentiment(self, content: str) -> Dict[str, Any]:
        try:
            result = await sentiment_analysis.ainvoke({"text": content})
            if isinstance(result, dict):
                return {
                    "label": str(result.get("label", "neutral")),
                    "score": float(result.get("score", 0.5)),
                    "aspects": result.get("aspects", {}),
                }
        except Exception as exc:
            logger.warning("sentiment tool failed, fallback to llm/keywords: %s", exc)

        llm_result = await self._llm_sentiment_analysis(content)
        if llm_result:
            return llm_result
        return self._keyword_sentiment(content)

    async def _llm_sentiment_analysis(self, content: str) -> Dict[str, Any]:
        try:
            prompt = ChatPromptTemplate.from_template(
                """请判断文本情感，仅输出 JSON:
{{"label": "positive/neutral/negative", "score": 0-1, "reason": "一句话原因"}}

文本:
{text}
"""
            )
            chain = prompt | self.llm
            response = await chain.ainvoke({"text": content[:2000]})
            response_text = self._response_to_text(response)
            match = re.search(r"\{[\s\S]*\}", response_text)
            if not match:
                return {}
            parsed = re.sub(r"\s+", " ", match.group(0)).strip()
            import json

            try:
                data = json.loads(parsed)
            except Exception:
                data = json.loads(parsed.replace("'", '"'))
            label = str(data.get("label", "neutral"))
            score = float(data.get("score", 0.5))
            return {"label": label, "score": max(0.0, min(1.0, score)), "aspects": {}}
        except Exception:
            return {}

    def _keyword_sentiment(self, text: str) -> Dict[str, Any]:
        positive_words = ["利好", "增长", "上涨", "优秀", "突破", "成功", "盈利", "预喜"]
        negative_words = ["利空", "下跌", "亏损", "违规", "处罚", "风险", "立案", "调查", "诉讼"]

        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)

        if pos_count > neg_count:
            return {"label": "positive", "score": min(0.9, 0.55 + pos_count * 0.08), "aspects": {}}
        if neg_count > pos_count:
            return {"label": "negative", "score": min(0.9, 0.55 + neg_count * 0.08), "aspects": {}}
        return {"label": "neutral", "score": 0.5, "aspects": {}}

    async def _summarize(self, content: str) -> str:
        if len(content) <= 200:
            return content
        try:
            prompt = ChatPromptTemplate.from_template(
                "请在50字以内总结文本核心信息：\n\n{text}"
            )
            chain = prompt | self.llm
            response = await chain.ainvoke({"text": content[:1500]})
            summary = self._response_to_text(response).strip()
            return summary or content[:80]
        except Exception as exc:
            logger.warning("summary failed: %s", exc)
            return content[:80]

    async def _extract_keywords(self, content: str) -> List[str]:
        try:
            prompt = ChatPromptTemplate.from_template(
                "从文本中提取3到5个关键词，使用逗号分隔：\n\n{text}"
            )
            chain = prompt | self.llm
            response = await chain.ainvoke({"text": content[:1200]})
            text = self._response_to_text(response)
            for sep in ["，", "、", "\n", ";", "；"]:
                text = text.replace(sep, ",")
            words = [item.strip(" -") for item in text.split(",") if item.strip(" -")]
            if words:
                return words[:5]
        except Exception as exc:
            logger.warning("keyword extraction failed: %s", exc)
        return self._fallback_keywords(content)

    def _fallback_keywords(self, content: str) -> List[str]:
        candidates = [
            "处罚",
            "调查",
            "诉讼",
            "违规",
            "盈利",
            "亏损",
            "增长",
            "下跌",
            "合作",
            "融资",
            "并购",
            "高管",
            "公告",
            "风险",
            "业绩",
        ]
        return [word for word in candidates if word in content][:5]

    def _response_to_text(self, response: Any) -> str:
        if response is None:
            return ""
        if hasattr(response, "content"):
            return str(response.content)
        return str(response)


def create_text_analysis_agent(
    llm: BaseLanguageModel,
    tools: Optional[List[BaseTool]] = None,
) -> Any:
    """
    Create a simplified ReAct text-analysis agent.
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

    default_tools = [sentiment_analysis, entity_recognition, event_extraction]
    all_tools = default_tools + (tools or [])
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(
        llm=llm,
        tools=all_tools,
        prompt=prompt,
    )
    return AgentExecutor(
        agent=agent,
        tools=all_tools,
        verbose=True,
        max_iterations=15,
        handle_parsing_errors=True,
    )
