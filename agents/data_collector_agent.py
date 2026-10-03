"""
Data collection agent for listed-company sentiment analysis.

This module focuses on:
- company entity normalization (name/alias/stock code)
- multi-source retrieval (news/announcement/social/forum)
- text cleaning and deduplication
- resilient fallback when external search is unavailable
"""
import hashlib
import logging
import re
import ast
import json
from urllib.parse import urlparse
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from .base_agent import BaseAgent

try:
    from ..tools import (
        entity_recognition,
        news_search,
        stock_open_company_info,
        web_crawl,
    )
except ImportError:  # pragma: no cover - script mode fallback
    from tools import (
        entity_recognition,
        news_search,
        stock_open_company_info,
        web_crawl,
    )

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


class DataCollectionAgent(BaseAgent):
    """
    Data collection agent.

    The input can be only `company_name`; stock code and aliases are inferred
    from an internal dictionary when possible.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[Any] = None,
        config: Optional[Any] = None,
    ):
        default_tools = [news_search, web_crawl, stock_open_company_info, entity_recognition]
        super().__init__(
            name="data_collector",
            llm=llm,
            tools=_unique_tools(default_tools + (tools or [])),
            memory=memory,
            config=config,
        )
        runtime_config = config if isinstance(config, dict) else {}
        self._min_strong_related_docs = self._bound_int(
            runtime_config.get(
                "min_distinct_docs",
                runtime_config.get("min_strong_related_docs", 6),
            ),
            default=6,
            min_value=1,
            max_value=200,
        )
        self._strong_relevance_floor = self._bound_float(
            runtime_config.get("strong_relevance_floor", 0.65),
            default=0.65,
            min_value=0.3,
            max_value=0.95,
        )
        self._supplemental_query_limit = self._bound_int(
            runtime_config.get("supplemental_query_limit", 8),
            default=8,
            min_value=1,
            max_value=20,
        )
        self._max_crawl_documents = self._bound_int(
            runtime_config.get("max_crawl_documents", 12),
            default=12,
            min_value=0,
            max_value=50,
        )
        self._max_final_documents = self._bound_int(
            runtime_config.get("max_final_documents", 30),
            default=30,
            min_value=1,
            max_value=200,
        )
        self._allow_synthetic_fallback = bool(runtime_config.get("allow_synthetic_fallback", False))
        self._prefer_deterministic_collection = bool(
            runtime_config.get("prefer_deterministic_collection", True)
        )
        self._collection_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._company_directory = self._build_company_directory()
        self._react_collector = self._build_react_collector()

    async def execute(
        self,
        company_name: str,
        stock_code: Optional[str] = None,
        sources: Optional[List[str]] = None,
        date_range: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self.state.task = f"collect sentiment data for {company_name}"
        self.state.status = "running"

        selected_sources = sources or ["news", "announcement", "social", "forum"]
        normalized = self._normalize_company(company_name, stock_code)
        normalized = await self._bootstrap_company_context(normalized)
        collection_strategy = self._decide_collection_strategy(
            resolved_company=normalized,
            sources=selected_sources,
        )
        start_date, end_date = self._resolve_date_range(date_range)

        cache_key = self._build_cache_key(
            company_name=normalized["company_name"],
            stock_code=normalized.get("stock_code"),
            sources=selected_sources,
            date_range=(start_date, end_date),
        )
        if cache_key in self._collection_cache:
            cached = self._collection_cache[cache_key]
            cached = self._filter_strong_related_documents(
                documents=cached,
                company_name=normalized["company_name"],
                stock_code=normalized.get("stock_code"),
                aliases=normalized.get("aliases", []),
            )
            cached = await self._ensure_minimum_strong_related_documents(
                documents=cached,
                company_name=normalized["company_name"],
                stock_code=normalized.get("stock_code"),
                aliases=normalized.get("aliases", []),
                sources=selected_sources,
                date_range=(start_date, end_date),
                entity_basis=normalized.get("entity_basis", []),
                strategy=collection_strategy,
            )
            cached.sort(
                key=lambda x: (x.get("relevance_score", 0.0), x.get("publish_date", "")),
                reverse=True,
            )
            cached = cached[: self._max_final_documents]
            self._collection_cache[cache_key] = cached
            self.state.status = "completed"
            self.state.output = {
                "documents": cached,
                "sources_used": selected_sources,
                "total_count": len(cached),
                "date_range": {"start": start_date, "end": end_date},
                "resolved_company": normalized,
                "collection_strategy": collection_strategy,
                "used_fallback_data": any(
                    doc.get("source") == "local_fallback" for doc in cached
                ),
            }
            return self.state.output

        logger.info(
            "collect start company=%s stock=%s sources=%s range=%s~%s",
            normalized["company_name"],
            normalized.get("stock_code"),
            selected_sources,
            start_date,
            end_date,
        )

        documents = await self._collect_data_with_react(
            company_name=normalized["company_name"],
            stock_code=normalized.get("stock_code"),
            aliases=normalized.get("aliases", []),
            sources=selected_sources,
            date_range=(start_date, end_date),
            strategy=collection_strategy,
            company_profile=normalized.get("company_profile"),
        )
        documents = self._deduplicate(documents)
        documents = await self._enhance_documents_with_crawl(documents)
        documents = self._deduplicate(documents)
        documents = await self._enrich_documents(
            documents=documents,
            company_name=normalized["company_name"],
            stock_code=normalized.get("stock_code"),
            aliases=normalized.get("aliases", []),
            entity_basis=normalized.get("entity_basis", []),
        )
        documents = self._filter_strong_related_documents(
            documents=documents,
            company_name=normalized["company_name"],
            stock_code=normalized.get("stock_code"),
            aliases=normalized.get("aliases", []),
        )

        if not documents:
            logger.warning("external sources returned no data, fallback to local template")
            documents = self._build_fallback_documents(
                company_name=normalized["company_name"],
                stock_code=normalized.get("stock_code"),
                date_range=(start_date, end_date),
                sources=selected_sources,
            )
            documents = await self._enrich_documents(
                documents=documents,
                company_name=normalized["company_name"],
                stock_code=normalized.get("stock_code"),
                aliases=normalized.get("aliases", []),
                entity_basis=normalized.get("entity_basis", []),
            )
            documents = self._filter_strong_related_documents(
                documents=documents,
                company_name=normalized["company_name"],
                stock_code=normalized.get("stock_code"),
                aliases=normalized.get("aliases", []),
            )

        documents = await self._ensure_minimum_strong_related_documents(
            documents=documents,
            company_name=normalized["company_name"],
            stock_code=normalized.get("stock_code"),
            aliases=normalized.get("aliases", []),
            sources=selected_sources,
            date_range=(start_date, end_date),
            entity_basis=normalized.get("entity_basis", []),
            strategy=collection_strategy,
        )
        documents = self._filter_strong_related_documents(
            documents=documents,
            company_name=normalized["company_name"],
            stock_code=normalized.get("stock_code"),
            aliases=normalized.get("aliases", []),
        )
        documents.sort(
            key=lambda x: (x.get("relevance_score", 0.0), x.get("publish_date", "")),
            reverse=True,
        )
        documents = documents[: self._max_final_documents]
        self._collection_cache[cache_key] = documents
        self.state.status = "completed"
        self.state.output = {
            "documents": documents,
            "sources_used": selected_sources,
            "total_count": len(documents),
            "date_range": {"start": start_date, "end": end_date},
            "resolved_company": normalized,
            "collection_strategy": collection_strategy,
            "used_fallback_data": any(
                doc.get("source") == "local_fallback" for doc in documents
            ),
        }
        return self.state.output

    def _build_company_directory(self) -> Dict[str, Dict[str, Any]]:
        profiles = [
            {
                "name": "贵州茅台",
                "stock_code": "600519",
                "aliases": ["茅台", "茅台股份", "贵州茅台股份有限公司"],
            },
            {
                "name": "宁德时代",
                "stock_code": "300750",
                "aliases": ["宁王", "宁德时代新能源", "宁德时代新能源科技股份有限公司"],
            },
            {
                "name": "中国平安",
                "stock_code": "601318",
                "aliases": ["平安", "平安集团", "中国平安保险"],
            },
            {
                "name": "比亚迪",
                "stock_code": "002594",
                "aliases": ["BYD", "比亚迪股份"],
            },
            {
                "name": "招商银行",
                "stock_code": "600036",
                "aliases": ["招行", "招商银行股份有限公司"],
            },
            {
                "name": "格力电器",
                "stock_code": "000651",
                "aliases": ["格力", "珠海格力电器"],
            },
            {
                "name": "海康威视",
                "stock_code": "002415",
                "aliases": ["海康", "杭州海康威视"],
            },
        ]

        index: Dict[str, Dict[str, Any]] = {}
        for profile in profiles:
            aliases = [profile["name"], profile["stock_code"], *profile.get("aliases", [])]
            for alias in aliases:
                key = alias.strip().lower()
                if key:
                    index[key] = profile
        return index

    def _normalize_company(
        self,
        company_name: str,
        stock_code: Optional[str],
    ) -> Dict[str, Any]:
        raw_name = (company_name or "").strip()
        raw_stock = (stock_code or "").strip() or None

        if re.fullmatch(r"\d{6}", raw_name) and not raw_stock:
            raw_stock = raw_name

        profile = self._company_directory.get(raw_name.lower()) if raw_name else None
        if not profile and raw_stock:
            profile = self._company_directory.get(raw_stock.lower())

        if profile:
            canonical_name = profile["name"]
            canonical_stock = raw_stock or profile.get("stock_code")
            aliases = [canonical_name, canonical_stock, *profile.get("aliases", [])]
        else:
            canonical_name = raw_name
            canonical_stock = raw_stock
            aliases = [canonical_name, canonical_stock] if canonical_stock else [canonical_name]

        deduped_aliases = [alias for alias in dict.fromkeys([a for a in aliases if a])]
        return {
            "company_name": canonical_name,
            "stock_code": canonical_stock,
            "aliases": deduped_aliases,
        }

    async def _bootstrap_company_context(
        self,
        normalized: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Always try stock_open_api first (when stock code is available),
        and use returned profile as the company-entity basis.
        """
        company_name = str(normalized.get("company_name") or "").strip()
        stock_code = str(normalized.get("stock_code") or "").strip()
        aliases = [str(x).strip() for x in (normalized.get("aliases") or []) if str(x).strip()]

        company_profile: Dict[str, Any] = {}
        if stock_code:
            try:
                profile = await stock_open_company_info.ainvoke(
                    {"stock_code": stock_code, "market": "auto"}
                )
            except Exception as exc:
                logger.warning("stock_open_company_info bootstrap failed: %s", exc)
                profile = {}

            if isinstance(profile, dict) and profile:
                company_profile = profile
                company_name = str(profile.get("company_name") or company_name).strip() or company_name
                stock_code = str(profile.get("stock_code") or stock_code).strip() or stock_code
                aliases = self._merge_aliases(
                    aliases,
                    self._extract_aliases_from_profile(profile),
                    [company_name, stock_code],
                )
            else:
                aliases = self._merge_aliases(aliases, [company_name, stock_code])
        else:
            aliases = self._merge_aliases(aliases, [company_name])

        entity_basis = self._build_entity_basis(
            company_name=company_name,
            stock_code=stock_code or None,
            aliases=aliases,
            company_profile=company_profile,
        )

        return {
            "company_name": company_name,
            "stock_code": stock_code or None,
            "aliases": aliases,
            "company_profile": company_profile,
            "entity_basis": entity_basis,
        }

    def _decide_collection_strategy(
        self,
        resolved_company: Dict[str, Any],
        sources: List[str],
    ) -> Dict[str, Any]:
        """
        Rule-based strategy so the agent can decide tool usage after
        company profile bootstrap.
        """
        company_profile = (
            resolved_company.get("company_profile")
            if isinstance(resolved_company.get("company_profile"), dict)
            else {}
        )
        has_profile = bool(company_profile)
        has_stock_code = bool(resolved_company.get("stock_code"))
        website_domain = self._extract_domain(str(company_profile.get("website") or ""))
        domain_filters = self._build_domain_filters(
            website_domain=website_domain,
            sources=sources,
        )
        strategy = {
            "news_primary_tool": "news_search",
            "announcement_include_profile_doc": has_profile and has_stock_code,
            "enabled_sources": [src for src in sources if src],
            "entity_basis_ready": bool(resolved_company.get("entity_basis")),
            "strict_company_match": True,
            "domain_filters": domain_filters,
        }
        return strategy

    def _extract_domain(self, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if not re.match(r"^https?://", text, flags=re.IGNORECASE):
            text = f"https://{text.lstrip('/')}"
        try:
            host = urlparse(text).netloc.lower()
        except Exception:
            return ""
        host = host.split("@")[-1].split(":")[0]
        if host.startswith("www."):
            host = host[4:]
        return host

    def _build_domain_filters(
        self,
        website_domain: str,
        sources: List[str],
    ) -> Dict[str, Dict[str, List[str]]]:
        source_set = {str(item).strip().lower() for item in (sources or []) if str(item).strip()}
        if not source_set:
            source_set = {"news", "announcement", "social", "forum"}

        official_domains = [
            "cninfo.com.cn",
            "static.cninfo.com.cn",
            "sse.com.cn",
            "query.sse.com.cn",
            "szse.cn",
            "disclosure.szse.cn",
            "neeq.com.cn",
        ]
        if website_domain and website_domain not in official_domains:
            official_domains.append(website_domain)

        finance_news_domains = [
            "eastmoney.com",
            "cnstock.com",
            "stcn.com",
            "10jqka.com.cn",
            "jrj.com.cn",
            "caixin.com",
            "yicai.com",
            "cs.com.cn",
        ]
        social_domains = ["weibo.com", "xueqiu.com", "guba.eastmoney.com"]
        exclude_domains = [
            "baike.baidu.com",
            "wikipedia.org",
            "zhihu.com",
            "tieba.baidu.com",
            "bilibili.com",
            "youtube.com",
        ]

        domain_filters: Dict[str, Dict[str, List[str]]] = {}
        if "announcement" in source_set:
            domain_filters["announcement"] = {
                "include_domains": official_domains,
                "exclude_domains": exclude_domains,
            }
        if "news" in source_set:
            domain_filters["news"] = {
                "include_domains": self._merge_aliases(finance_news_domains, official_domains),
                "exclude_domains": exclude_domains,
            }
        if "social" in source_set:
            domain_filters["social"] = {
                "include_domains": social_domains,
                "exclude_domains": exclude_domains,
            }
        if "forum" in source_set:
            domain_filters["forum"] = {
                "include_domains": ["xueqiu.com", "guba.eastmoney.com"],
                "exclude_domains": exclude_domains,
            }
        return domain_filters

    def _merge_aliases(self, *groups: List[str]) -> List[str]:
        merged: List[str] = []
        seen = set()
        for group in groups:
            for item in group or []:
                value = str(item or "").strip()
                if not value:
                    continue
                key = value.lower()
                if key in seen:
                    continue
                seen.add(key)
                merged.append(value)
        return merged

    def _extract_aliases_from_profile(self, profile: Dict[str, Any]) -> List[str]:
        raw = profile.get("raw") if isinstance(profile.get("raw"), dict) else {}
        candidates = [
            profile.get("company_name"),
            profile.get("stock_code"),
            profile.get("full_code"),
            raw.get("A股简称"),
            raw.get("A股扩位简称"),
            raw.get("曾用名"),
            raw.get("英文名称"),
        ]

        aliases: List[str] = []
        for value in candidates:
            text = str(value or "").strip()
            if not text:
                continue
            parts = re.split(r"[;,，、/\s]+", text)
            for part in parts:
                token = part.strip()
                if token:
                    aliases.append(token)
        return self._merge_aliases(aliases)

    def _build_entity_basis(
        self,
        company_name: str,
        stock_code: Optional[str],
        aliases: List[str],
        company_profile: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        basis: List[Dict[str, Any]] = []

        if company_name:
            basis.append(
                {
                    "text": company_name,
                    "type": "COMPANY",
                    "confidence": 0.99,
                    "source": "stock_open_api" if company_profile else "input",
                }
            )
        if stock_code:
            basis.append(
                {
                    "text": stock_code,
                    "type": "STOCK_CODE",
                    "confidence": 0.99,
                    "source": "stock_open_api" if company_profile else "input",
                }
            )

        for alias in aliases:
            if alias in {company_name, stock_code}:
                continue
            basis.append(
                {
                    "text": alias,
                    "type": "COMPANY_ALIAS",
                    "confidence": 0.95,
                    "source": "stock_open_api" if company_profile else "input",
                }
            )

        unique_basis: List[Dict[str, Any]] = []
        seen = set()
        for item in basis:
            key = (item.get("text"), item.get("type"))
            if key in seen:
                continue
            seen.add(key)
            unique_basis.append(item)
        return unique_basis

    def _resolve_date_range(self, date_range: Optional[Any]) -> Tuple[str, str]:
        if isinstance(date_range, Sequence) and not isinstance(date_range, str):
            values = list(date_range)
            if len(values) >= 2:
                start = self._to_date_str(values[0])
                end = self._to_date_str(values[1])
                return (start, end) if start <= end else (end, start)

        if isinstance(date_range, str):
            text = date_range.strip()
            exact_dates = re.findall(r"\d{4}-\d{1,2}-\d{1,2}", text)
            if len(exact_dates) >= 2:
                start = self._to_date_str(exact_dates[0])
                end = self._to_date_str(exact_dates[1])
                return (start, end) if start <= end else (end, start)
            if len(exact_dates) == 1:
                day = self._to_date_str(exact_dates[0])
                return (day, day)

            if any(token in text.lower() for token in ["最近", "近", "latest", "past", "last"]):
                matched = re.search(r"(\d+)", text)
                days = int(matched.group(1)) if matched else 7
                days = max(1, min(days, 90))
                end = datetime.now().date()
                start = end - timedelta(days=days - 1)
                return (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

        end = datetime.now().date()
        start = end - timedelta(days=6)
        return (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    def _to_date_str(self, value: Any) -> str:
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d")
        if isinstance(value, date):
            return value.strftime("%Y-%m-%d")
        text = str(value).strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            return text
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return parsed.strftime("%Y-%m-%d")
        except Exception:
            pass
        return datetime.now().strftime("%Y-%m-%d")

    def _build_cache_key(
        self,
        company_name: str,
        stock_code: Optional[str],
        sources: List[str],
        date_range: Tuple[str, str],
    ) -> str:
        raw = "|".join(
            [
                company_name,
                stock_code or "",
                ",".join(sorted(sources)),
                date_range[0],
                date_range[1],
            ]
        )
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _build_source_queries(
        self,
        company_name: str,
        stock_code: Optional[str],
        aliases: List[str],
    ) -> Dict[str, str]:
        alias_candidates = [
            token for token in aliases
            if token and token not in {company_name, stock_code}
        ][:2]
        quoted_name = f"\"{company_name}\"" if company_name else ""
        stock_anchor = stock_code or ""
        alias_expr = " OR ".join([f"\"{token}\"" for token in alias_candidates])

        base_parts = [part for part in [quoted_name, stock_anchor] if part]
        base_expr = " ".join(base_parts).strip() or company_name

        news_query = f"{base_expr} 公告 业绩 处罚 问询"
        announcement_query = f"{base_expr} 公告 临时公告 年报"
        social_query = f"{base_expr} 微博 舆情 投资者"
        forum_query = f"{base_expr} 雪球 股吧 论坛"

        if alias_expr:
            news_query = f"{news_query} ({alias_expr})"
            social_query = f"{social_query} ({alias_expr})"

        return {
            "news": news_query.strip(),
            "announcement": announcement_query.strip(),
            "social": social_query.strip(),
            "forum": forum_query.strip(),
        }

    def _bound_int(
        self,
        value: Any,
        default: int,
        min_value: int,
        max_value: int,
    ) -> int:
        try:
            parsed = int(value)
        except Exception:
            parsed = default
        return max(min_value, min(max_value, parsed))

    def _bound_float(
        self,
        value: Any,
        default: float,
        min_value: float,
        max_value: float,
    ) -> float:
        try:
            parsed = float(value)
        except Exception:
            parsed = default
        return max(min_value, min(max_value, parsed))

    def _prepare_company_terms(
        self,
        company_name: str,
        stock_code: Optional[str],
        aliases: List[str],
    ) -> List[str]:
        terms = [company_name, stock_code, *aliases]
        prepared: List[str] = []
        seen = set()
        for term in terms:
            text = str(term or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            prepared.append(text)
        return prepared

    def _normalize_company_term(self, text: str) -> str:
        normalized = str(text or "").strip().lower()
        if not normalized:
            return ""
        return re.sub(r"[\s\-_.,;:!?\"'`~()\[\]{}<>/\\|]+", "", normalized)

    def _count_company_term_hits(
        self,
        title: str,
        content: str,
        terms: List[str],
    ) -> Tuple[int, int, int]:
        raw_title = str(title or "").lower()
        raw_content = str(content or "").lower()
        normalized_title = self._normalize_company_term(raw_title)
        normalized_content = self._normalize_company_term(raw_content)

        title_hits = set()
        content_hits = set()
        for term in terms:
            normalized_term = self._normalize_company_term(term)
            if not normalized_term:
                continue

            if normalized_term.isdigit():
                hit_title = normalized_term in raw_title
                hit_content = normalized_term in raw_content
            else:
                hit_title = normalized_term in normalized_title
                hit_content = normalized_term in normalized_content

            if hit_title:
                title_hits.add(normalized_term)
            if hit_content:
                content_hits.add(normalized_term)

        total_hits = len(title_hits | content_hits)
        return len(title_hits), len(content_hits), total_hits

    def _build_diversity_key(self, doc: Dict[str, Any]) -> str:
        url = str(doc.get("url", "") or "").strip().lower()
        publish_date = self._to_date_str(doc.get("publish_date") or datetime.now())
        title = self._normalize_company_term(self._clean_text(doc.get("title", "")))
        content = self._normalize_company_term(self._clean_text(doc.get("content", "")))

        if url and not url.startswith("local://"):
            return f"url:{url}|date:{publish_date}"

        seed = "|".join(
            [
                str(doc.get("type", "")).strip().lower(),
                publish_date,
                title[:180],
                content[:360],
            ]
        )
        return f"sem:{hashlib.md5(seed.encode('utf-8')).hexdigest()}"

    def _deduplicate_for_diversity(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        unique_docs: List[Dict[str, Any]] = []
        for doc in documents:
            key = self._build_diversity_key(doc)
            if key in seen:
                continue
            seen.add(key)
            unique_docs.append(doc)
        return unique_docs

    async def _enhance_documents_with_crawl(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not documents or self._max_crawl_documents <= 0:
            return documents

        candidate_indexes: List[int] = []
        for idx, doc in enumerate(documents):
            if len(candidate_indexes) >= self._max_crawl_documents:
                break
            source = str(doc.get("source") or "").strip().lower()
            doc_type = str(doc.get("type") or "").strip().lower()
            url = str(doc.get("url") or "").strip()
            if source in {"stock_open_api", "local_fallback"}:
                continue
            if doc_type not in {"news", "announcement"}:
                continue
            if not re.match(r"^https?://", url, flags=re.IGNORECASE):
                continue
            candidate_indexes.append(idx)

        if not candidate_indexes:
            return documents

        import asyncio

        tasks = [
            web_crawl.ainvoke(
                {
                    "url": str(documents[idx].get("url") or "").strip(),
                    "max_chars": 8000,
                    "include_links": False,
                    "timeout": 18,
                }
            )
            for idx in candidate_indexes
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        enhanced = [dict(doc) for doc in documents]
        crawl_success = 0
        for idx, result in zip(candidate_indexes, results):
            if not isinstance(result, dict):
                continue

            original_doc = enhanced[idx]
            metadata = dict(original_doc.get("metadata") or {})
            unreadable_reason = str(
                result.get("unreadable_reason")
                or self._detect_unreadable_reason(result.get("content", ""))
                or ""
            ).strip()
            if result.get("unreadable") or unreadable_reason:
                metadata.update(
                    {
                        "crawl_enhanced": False,
                        "crawl_url": result.get("final_url") or original_doc.get("url", ""),
                        "crawl_source": result.get("source", ""),
                        "crawl_status": result.get("status_code", ""),
                        "crawl_unreadable": True,
                        "crawl_unreadable_reason": unreadable_reason or "unreadable_crawled_content",
                    }
                )
                original_doc["metadata"] = metadata
                continue

            crawled_content = self._clean_text(result.get("content", ""))
            crawled_title = self._clean_text(result.get("title", ""))
            if len(crawled_content) < 80 and not crawled_title:
                continue

            original_content = self._clean_text(original_doc.get("content", ""))
            metadata.update(
                {
                    "crawl_enhanced": True,
                    "crawl_url": result.get("final_url") or original_doc.get("url", ""),
                    "crawl_source": result.get("source", ""),
                    "crawl_status": result.get("status_code", ""),
                }
            )

            if len(crawled_content) >= max(120, len(original_content)):
                original_doc["content"] = crawled_content
            if crawled_title and len(crawled_title) >= len(str(original_doc.get("title") or "").strip()):
                original_doc["title"] = crawled_title
            if result.get("source"):
                original_doc["source"] = result.get("source")
            original_doc["metadata"] = metadata
            crawl_success += 1

        logger.info(
            "crawl enhancement finished: attempted=%s success=%s",
            len(candidate_indexes),
            crawl_success,
        )
        return enhanced

    def _filter_strong_related_documents(
        self,
        documents: List[Dict[str, Any]],
        company_name: str,
        stock_code: Optional[str],
        aliases: List[str],
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []

        kept: List[Dict[str, Any]] = []
        dropped = 0
        for doc in documents:
            source = str(doc.get("source") or "").strip().lower()
            if source in {"stock_open_api", "local_fallback"}:
                kept.append(doc)
                continue
            if self._is_strong_company_related(
                doc=doc,
                company_name=company_name,
                stock_code=stock_code,
                aliases=aliases,
            ):
                kept.append(doc)
            else:
                dropped += 1

        if dropped > 0:
            logger.info(
                "precision filter removed unrelated documents: kept=%s dropped=%s",
                len(kept),
                dropped,
            )
        return kept

    def _is_strong_company_related(
        self,
        doc: Dict[str, Any],
        company_name: str,
        stock_code: Optional[str],
        aliases: List[str],
    ) -> bool:
        metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
        if isinstance(metadata, dict) and metadata.get("force_strong_related"):
            return True
        if isinstance(metadata, dict) and metadata.get("unreadable_content"):
            return False

        title = str(doc.get("title", "") or "")
        content = str(doc.get("content", "") or "")
        terms = self._prepare_company_terms(company_name, stock_code, aliases)
        title_hits, _, total_hits = self._count_company_term_hits(title, content, terms)
        primary_terms = [token for token in [company_name, stock_code] if token]
        _, _, primary_total_hits = self._count_company_term_hits(title, content, primary_terms)
        relevance = self._bound_float(
            doc.get("relevance_score", 0.0),
            default=0.0,
            min_value=0.0,
            max_value=1.0,
        )

        if primary_total_hits >= 1 and (title_hits >= 1 or relevance >= 0.55):
            return True
        if title_hits >= 1 and total_hits >= 2:
            return True
        if total_hits >= 2 and relevance >= 0.62:
            return True
        if total_hits >= 1 and relevance >= max(self._strong_relevance_floor, 0.75):
            return True
        return False

    def _count_strong_related_documents(
        self,
        documents: List[Dict[str, Any]],
        company_name: str,
        stock_code: Optional[str],
        aliases: List[str],
    ) -> int:
        return sum(
            1
            for doc in documents
            if self._is_strong_company_related(
                doc=doc,
                company_name=company_name,
                stock_code=stock_code,
                aliases=aliases,
            )
        )

    def _build_targeted_source_queries(
        self,
        company_name: str,
        stock_code: Optional[str],
        aliases: List[str],
        sources: List[str],
    ) -> List[Tuple[str, str]]:
        source_set = {str(item).strip().lower() for item in (sources or []) if str(item).strip()}
        if not source_set:
            source_set = {"news", "announcement", "social", "forum"}

        alias_candidates = [
            alias for alias in aliases
            if alias and alias not in {company_name, stock_code}
        ][:2]
        stock_anchor = stock_code or company_name

        query_specs: List[Tuple[str, str]] = []
        if "news" in source_set:
            query_specs.extend(
                [
                    ("news", f"\"{company_name}\""),
                    ("news", f"\"{company_name}\" {stock_anchor}"),
                    ("news", f"\"{company_name}\" {stock_anchor} 公告 业绩"),
                    ("news", f"\"{company_name}\" {stock_anchor} 处罚 调查 诉讼"),
                ]
            )
            for alias in alias_candidates:
                query_specs.append(("news", f"\"{alias}\" {stock_anchor}"))
        if "announcement" in source_set:
            query_specs.extend(
                [
                    ("announcement", f"{stock_anchor} 公告 cninfo"),
                    ("announcement", f"{stock_anchor} 公告 sse szse"),
                    ("announcement", f"\"{company_name}\" {stock_anchor} 临时公告"),
                    ("announcement", f"\"{company_name}\" {stock_anchor} 年报 问询函"),
                ]
            )
        if "social" in source_set:
            query_specs.extend(
                [
                    ("social", f"\"{company_name}\" {stock_anchor} 舆情"),
                    ("social", f"\"{company_name}\" {stock_anchor} 微博"),
                    ("social", f"\"{company_name}\" {stock_anchor} 投资者讨论"),
                ]
            )
        if "forum" in source_set:
            query_specs.extend(
                [
                    ("forum", f"\"{company_name}\" {stock_anchor} 雪球"),
                    ("forum", f"\"{company_name}\" {stock_anchor} 股吧"),
                    ("forum", f"\"{company_name}\" {stock_anchor} 论坛"),
                ]
            )

        deduped: List[Tuple[str, str]] = []
        seen = set()
        for source_type, query in query_specs:
            key = (source_type.strip().lower(), query.strip().lower())
            if not key[1] or key in seen:
                continue
            seen.add(key)
            deduped.append((source_type, query.strip()))
        return deduped

    async def _collect_additional_documents_for_strong_relevance(
        self,
        company_name: str,
        stock_code: Optional[str],
        aliases: List[str],
        sources: List[str],
        date_range: Tuple[str, str],
        missing_count: int,
        strategy: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        import asyncio

        if missing_count <= 0:
            return []

        query_specs = self._build_targeted_source_queries(
            company_name=company_name,
            stock_code=stock_code,
            aliases=aliases,
            sources=sources,
        )
        if not query_specs:
            return []

        per_query_limit = max(8, min(20, missing_count * 2))
        planned_queries = query_specs[: self._supplemental_query_limit]
        tasks = [
            self._query_source(
                source_type=source_type,
                query=query,
                date_range=date_range,
                limit=per_query_limit,
                strategy=strategy,
            )
            for source_type, query in planned_queries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        docs: List[Dict[str, Any]] = []
        for (source_type, query), result in zip(planned_queries, results):
            if isinstance(result, list):
                docs.extend(result)
            elif isinstance(result, Exception):
                logger.warning(
                    "supplemental query failed source=%s query=%s err=%s",
                    source_type,
                    query,
                    result,
                )
        return self._deduplicate_for_diversity(self._deduplicate(docs))

    def _build_strong_related_fallback_documents(
        self,
        company_name: str,
        stock_code: Optional[str],
        aliases: List[str],
        date_range: Tuple[str, str],
        sources: List[str],
        count: int,
    ) -> List[Dict[str, Any]]:
        if count <= 0:
            return []

        start_date, end_date = date_range
        code_text = f"({stock_code})" if stock_code else ""
        alias_candidates = [
            alias for alias in aliases
            if alias and alias not in {company_name, stock_code}
        ][:3]
        alias_text = ", ".join(alias_candidates)

        preferred_sources = [src for src in sources if src in {"news", "announcement", "social", "forum"}]
        if not preferred_sources:
            preferred_sources = ["news"]

        topic_templates = [
            ("announcement_tracking", "focus on disclosure and official notices"),
            ("earnings_update", "focus on revenue and profit trend"),
            ("cooperation_progress", "focus on signing and business collaboration"),
            ("regulatory_risk", "focus on inquiry, investigation and compliance"),
            ("litigation_progress", "focus on lawsuit and arbitration updates"),
            ("investor_interaction", "focus on investor Q&A and feedback"),
            ("capacity_expansion", "focus on expansion and production schedule"),
            ("industry_comparison", "focus on peer benchmark signals"),
            ("fund_flow", "focus on financing, buyback and reduction"),
            ("management_change", "focus on executive changes and governance"),
            ("market_reaction", "focus on market sentiment and hot topics"),
            ("operational_risk", "focus on supply chain and cost pressure"),
        ]

        end_dt = datetime.fromisoformat(self._to_date_str(end_date))
        start_dt = datetime.fromisoformat(self._to_date_str(start_date))
        if start_dt > end_dt:
            start_dt, end_dt = end_dt, start_dt
        span_days = max(0, (end_dt - start_dt).days)

        docs: List[Dict[str, Any]] = []
        for idx in range(count):
            source_type = preferred_sources[idx % len(preferred_sources)]
            topic, angle = topic_templates[idx % len(topic_templates)]
            if span_days > 0:
                offset = idx % (span_days + 1)
            else:
                offset = idx % 30
            publish_dt = end_dt - timedelta(days=offset)
            publish_date = publish_dt.strftime("%Y-%m-%d")

            title = f"{company_name}{code_text} {topic} supplemental#{idx + 1}"
            content_parts = [
                f"{company_name}{code_text} {topic} intelligence supplement.",
                f"analysis angle: {angle}.",
                f"date anchor: {publish_date}.",
            ]
            if stock_code:
                content_parts.append(f"stock anchor: {stock_code}.")
            if alias_text:
                content_parts.append(f"aliases: {alias_text}.")
            content_parts.append(f"sample_id: {idx + 1}.")

            docs.append(
                {
                    "type": source_type,
                    "title": title,
                    "url": f"local://strong-fallback/{company_name}/{publish_date}/{idx}",
                    "source": "local_fallback",
                    "publish_date": publish_date,
                    "content": " ".join(content_parts),
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {
                        "is_fallback": True,
                        "force_strong_related": True,
                        "fallback_reason": "insufficient_distinct_results",
                        "diversity_topic": topic,
                        "diversity_index": idx + 1,
                    },
                }
            )
        return self._deduplicate_for_diversity(docs)

    async def _ensure_minimum_strong_related_documents(
        self,
        documents: List[Dict[str, Any]],
        company_name: str,
        stock_code: Optional[str],
        aliases: List[str],
        sources: List[str],
        date_range: Tuple[str, str],
        entity_basis: Optional[List[Dict[str, Any]]] = None,
        strategy: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        required = max(1, int(self._min_strong_related_docs))
        documents = self._filter_strong_related_documents(
            documents=self._deduplicate_for_diversity(self._deduplicate(documents)),
            company_name=company_name,
            stock_code=stock_code,
            aliases=aliases,
        )
        strong_count = self._count_strong_related_documents(
            documents=documents,
            company_name=company_name,
            stock_code=stock_code,
            aliases=aliases,
        )
        if strong_count >= required:
            return documents

        missing = required - strong_count
        logger.info(
            "strong-related docs below threshold: current=%s required=%s; start supplemental search",
            strong_count,
            required,
        )
        supplemental_docs = await self._collect_additional_documents_for_strong_relevance(
            company_name=company_name,
            stock_code=stock_code,
            aliases=aliases,
            sources=sources,
            date_range=date_range,
            missing_count=missing,
            strategy=strategy,
        )
        if supplemental_docs:
            supplemental_docs = await self._enhance_documents_with_crawl(supplemental_docs)
            supplemental_enriched = await self._enrich_documents(
                documents=supplemental_docs,
                company_name=company_name,
                stock_code=stock_code,
                aliases=aliases,
                entity_basis=entity_basis or [],
            )
            documents = self._deduplicate_for_diversity(
                self._deduplicate(documents + supplemental_enriched)
            )
            documents = self._filter_strong_related_documents(
                documents=documents,
                company_name=company_name,
                stock_code=stock_code,
                aliases=aliases,
            )
            strong_count = self._count_strong_related_documents(
                documents=documents,
                company_name=company_name,
                stock_code=stock_code,
                aliases=aliases,
            )

        if strong_count >= required:
            return documents

        if self._allow_synthetic_fallback:
            fallback_docs = self._build_strong_related_fallback_documents(
                company_name=company_name,
                stock_code=stock_code,
                aliases=aliases,
                date_range=date_range,
                sources=sources,
                count=(required - strong_count),
            )
            if fallback_docs:
                fallback_enriched = await self._enrich_documents(
                    documents=fallback_docs,
                    company_name=company_name,
                    stock_code=stock_code,
                    aliases=aliases,
                    entity_basis=entity_basis or [],
                )
                documents = self._deduplicate_for_diversity(
                    self._deduplicate(documents + fallback_enriched)
                )
                documents = self._filter_strong_related_documents(
                    documents=documents,
                    company_name=company_name,
                    stock_code=stock_code,
                    aliases=aliases,
                )
                strong_count = self._count_strong_related_documents(
                    documents=documents,
                    company_name=company_name,
                    stock_code=stock_code,
                    aliases=aliases,
                )

        final_count = strong_count
        if final_count < required:
            logger.warning(
                "cannot fully satisfy strong-related minimum; final=%s required=%s",
                final_count,
                required,
            )
        else:
            logger.info("strong-related minimum satisfied: final=%s required=%s", final_count, required)
        return documents

    def _build_react_collector(self) -> Optional[Any]:
        """
        Build a ReAct-style tool-calling collector.

        Current LangChain in this workspace exposes `create_agent` instead of
        legacy `create_react_agent`. This still executes a tool-call loop where
        the model decides which tool to call.
        """
        try:
            from langchain.agents import create_agent
        except Exception as exc:
            logger.warning("langchain create_agent is unavailable: %s", exc)
            return None

        system_prompt = (
            "You are a data collection agent for listed-company sentiment analysis.\n"
            "You can call tools to fetch company profile and public news data.\n"
            "Use `web_crawl` when you need to fetch website content directly.\n"
            "For announcement/regulatory disclosure tasks in China A-share context, "
            "prioritize crawling CNINFO first: http://www.cninfo.com.cn/ .\n"
            "Choose tools autonomously based on the task.\n"
            "Always finish by returning JSON only with this schema:\n"
            "{\n"
            '  "documents": [\n'
            "    {\n"
            '      "type": "news|announcement|social|forum",\n'
            '      "title": "...",\n'
            '      "url": "...",\n'
            '      "source": "...",\n'
            '      "publish_date": "YYYY-MM-DD or ISO date",\n'
            '      "content": "..."\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Rules:\n"
            "- Use only available tools.\n"
            "- Prefer calling tools instead of fabricating content.\n"
            "- Use web_crawl for pages that search APIs cannot cover.\n"
            "- Keep documents relevant to the target company.\n"
            "- Return no markdown, no explanation, JSON only."
        )

        try:
            return create_agent(
                model=self.llm,
                tools=self.tools,
                system_prompt=system_prompt,
                debug=False,
                name="data_collector_react",
            )
        except Exception as exc:
            logger.warning("failed to initialize react collector: %s", exc)
            return None

    async def _collect_data_with_react(
        self,
        company_name: str,
        stock_code: Optional[str],
        aliases: List[str],
        sources: List[str],
        date_range: Tuple[str, str],
        strategy: Optional[Dict[str, Any]] = None,
        company_profile: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Precision-first collection path:
        1) deterministic stock/profile-driven Tavily retrieval
        2) optional ReAct fallback only when deterministic path fails
        """
        strategy = strategy or {}
        docs: List[Dict[str, Any]] = []

        if strategy.get("announcement_include_profile_doc") and company_profile:
            profile_doc = self._build_announcement_doc_from_company_info(
                info=company_profile,
                company_name=company_name,
                date_range=date_range,
            )
            if profile_doc:
                docs.append(profile_doc)

        if self._prefer_deterministic_collection:
            deterministic_docs = await self._collect_data(
                company_name=company_name,
                stock_code=stock_code,
                aliases=aliases,
                sources=sources,
                date_range=date_range,
                strategy=strategy,
                company_profile=company_profile,
            )
            if deterministic_docs:
                return docs + deterministic_docs
            logger.info("deterministic collection returned no documents, trying react fallback")

        if not self._react_collector:
            logger.warning("react collector is not ready; fallback to legacy collection")
            legacy = await self._collect_data(
                company_name=company_name,
                stock_code=stock_code,
                aliases=aliases,
                sources=sources,
                date_range=date_range,
                strategy=strategy,
                company_profile=company_profile,
            )
            return docs + legacy

        query_map = self._build_source_queries(company_name, stock_code, aliases)
        source_lines = []
        for src in sources:
            if src in query_map:
                source_lines.append(f"- {src}: {query_map[src]}")

        user_prompt = (
            "Collect sentiment-related documents for this listed company.\n"
            f"company_name: {company_name}\n"
            f"stock_code: {stock_code or ''}\n"
            f"aliases: {', '.join(aliases[:8])}\n"
            f"date_from: {date_range[0]}\n"
            f"date_to: {date_range[1]}\n"
            f"enabled_sources: {', '.join(sources)}\n"
            "preferred_disclosure_site: http://www.cninfo.com.cn/\n"
            "queries_by_source:\n"
            f"{chr(10).join(source_lines)}\n"
            "If announcement source is enabled, use web_crawl on cninfo-related pages first.\n"
            "Return final JSON only."
        )

        try:
            result = await self._react_collector.ainvoke(
                {"messages": [{"role": "user", "content": user_prompt}]},
                config={"recursion_limit": 30},
            )
            react_docs = self._parse_react_documents(result=result, enabled_sources=sources)
            if react_docs:
                return docs + react_docs
            logger.warning("react collector returned no parseable documents, fallback to legacy path")
        except Exception as exc:
            logger.warning("react collector failed, fallback to legacy path: %s", exc)

        legacy = await self._collect_data(
            company_name=company_name,
            stock_code=stock_code,
            aliases=aliases,
            sources=sources,
            date_range=date_range,
            strategy=strategy,
            company_profile=company_profile,
        )
        return docs + legacy

    def _parse_react_documents(
        self,
        result: Any,
        enabled_sources: List[str],
    ) -> List[Dict[str, Any]]:
        docs = self._extract_docs_from_final_ai_message(result)
        if not docs:
            docs = self._extract_docs_from_tool_messages(result)
        if not docs:
            return []

        now_ts = datetime.now().isoformat()
        allowed = {src.strip().lower() for src in enabled_sources if src}
        normalized_docs: List[Dict[str, Any]] = []
        for item in docs:
            if not isinstance(item, dict):
                continue

            source_type = str(item.get("type") or "news").strip().lower()
            if allowed and source_type not in allowed:
                continue

            normalized_docs.append(
                {
                    "type": source_type or "news",
                    "title": str(item.get("title") or "").strip(),
                    "url": str(item.get("url") or "").strip(),
                    "source": str(item.get("source") or "react_agent").strip(),
                    "publish_date": str(item.get("publish_date") or "").strip(),
                    "content": str(item.get("content") or "").strip(),
                    "timestamp": now_ts,
                }
            )

        return normalized_docs

    def _extract_docs_from_final_ai_message(self, result: Any) -> List[Dict[str, Any]]:
        messages = result.get("messages", []) if isinstance(result, dict) else []
        ai_text = ""
        for message in reversed(messages):
            mtype = getattr(message, "type", "")
            if mtype == "ai":
                ai_text = self._content_to_text(getattr(message, "content", ""))
                if ai_text:
                    break

        if not ai_text:
            return []

        payload = self._extract_json_payload(ai_text)
        if not isinstance(payload, dict):
            return []
        docs = payload.get("documents")
        return docs if isinstance(docs, list) else []

    def _extract_docs_from_tool_messages(self, result: Any) -> List[Dict[str, Any]]:
        messages = result.get("messages", []) if isinstance(result, dict) else []
        docs: List[Dict[str, Any]] = []
        for message in messages:
            mtype = getattr(message, "type", "")
            if mtype != "tool":
                continue
            tool_name = str(getattr(message, "name", "") or "").strip().lower()
            if tool_name not in {"news_search", "web_crawl"}:
                continue

            raw = self._content_to_text(getattr(message, "content", ""))
            parsed = self._extract_json_payload(raw)
            if parsed is None:
                try:
                    parsed = ast.literal_eval(raw)
                except Exception:
                    parsed = None
            if isinstance(parsed, dict):
                parsed = [parsed]
            if not isinstance(parsed, list):
                continue

            for item in parsed:
                if not isinstance(item, dict):
                    continue
                source = item.get("source", tool_name)
                url = item.get("url", "")
                source_text = str(source or "")
                url_text = str(url or "")
                doc_type = "announcement" if ("cninfo.com.cn" in source_text or "cninfo.com.cn" in url_text) else "news"
                publish_date = item.get("publish_date", "")
                if not publish_date:
                    fetched_at = str(item.get("fetched_at", "")).strip()
                    publish_date = fetched_at[:10] if len(fetched_at) >= 10 else ""
                docs.append(
                    {
                        "type": doc_type,
                        "title": item.get("title", ""),
                        "url": url,
                        "source": source,
                        "publish_date": publish_date,
                        "content": item.get("content", ""),
                    }
                )
        return docs

    def _content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                    continue
                if isinstance(block, dict):
                    if "text" in block:
                        parts.append(str(block.get("text") or ""))
                    elif block.get("type") == "text":
                        parts.append(str(block.get("text") or ""))
                    elif "content" in block:
                        parts.append(str(block.get("content") or ""))
            return "\n".join([p for p in parts if p]).strip()
        return str(content or "")

    def _extract_json_payload(self, text: str) -> Optional[Any]:
        raw = (text or "").strip()
        if not raw:
            return None

        candidates: List[str] = []
        fenced = re.findall(r"```json\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
        if fenced:
            candidates.extend([c.strip() for c in fenced if c.strip()])
        generic_fenced = re.findall(r"```([\s\S]*?)```", raw, flags=re.IGNORECASE)
        if generic_fenced:
            candidates.extend([c.strip() for c in generic_fenced if c.strip()])

        brace_match = re.search(r"\{[\s\S]*\}", raw)
        if brace_match:
            candidates.append(brace_match.group(0).strip())

        bracket_match = re.search(r"\[[\s\S]*\]", raw)
        if bracket_match:
            candidates.append(bracket_match.group(0).strip())

        candidates.append(raw)

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except Exception:
                continue
        return None

    async def _collect_data(
        self,
        company_name: str,
        stock_code: Optional[str],
        aliases: List[str],
        sources: List[str],
        date_range: Tuple[str, str],
        strategy: Optional[Dict[str, Any]] = None,
        company_profile: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        import asyncio

        strategy = strategy or {}
        query_map = self._build_source_queries(company_name, stock_code, aliases)
        tasks = []
        if "news" in sources:
            tasks.append(
                self._collect_news(
                    query=query_map["news"],
                    date_range=date_range,
                    strategy=strategy,
                )
            )
        if "announcement" in sources:
            tasks.append(
                self._collect_announcements(
                    query=query_map["announcement"],
                    date_range=date_range,
                    company_name=company_name,
                    stock_code=stock_code,
                    strategy=strategy,
                    company_profile=company_profile,
                )
            )
        if "social" in sources:
            tasks.append(
                self._collect_social_media(
                    query=query_map["social"],
                    date_range=date_range,
                    strategy=strategy,
                )
            )
        if "forum" in sources:
            tasks.append(
                self._collect_forums(
                    query=query_map["forum"],
                    date_range=date_range,
                    strategy=strategy,
                )
            )

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_documents: List[Dict[str, Any]] = []
        for result in results:
            if isinstance(result, list):
                all_documents.extend(result)
            elif isinstance(result, Exception):
                logger.warning("data source collection failed: %s", result)
        return all_documents

    async def _collect_news(
        self,
        query: str,
        date_range: Tuple[str, str],
        strategy: Optional[Dict[str, Any]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        return await self._query_source(
            source_type="news",
            query=query,
            date_range=date_range,
            limit=limit,
            strategy=strategy,
        )

    async def _collect_announcements(
        self,
        query: str,
        date_range: Tuple[str, str],
        company_name: str,
        stock_code: Optional[str],
        strategy: Optional[Dict[str, Any]] = None,
        company_profile: Optional[Dict[str, Any]] = None,
        limit: int = 12,
    ) -> List[Dict[str, Any]]:
        strategy = strategy or {}
        docs: List[Dict[str, Any]] = []

        if strategy.get("announcement_include_profile_doc") and company_profile:
            profile_doc = self._build_announcement_doc_from_company_info(
                info=company_profile,
                company_name=company_name,
                date_range=date_range,
            )
            if profile_doc:
                docs.append(profile_doc)

        fallback_limit = max(0, limit - len(docs))
        if fallback_limit > 0:
            docs.extend(
                await self._query_source(
                    source_type="announcement",
                    query=query,
                    date_range=date_range,
                    limit=fallback_limit,
                    strategy=strategy,
                )
            )
        return docs[:limit]

    def _build_announcement_doc_from_company_info(
        self,
        info: Dict[str, Any],
        company_name: str,
        date_range: Tuple[str, str],
    ) -> Dict[str, Any]:
        stock_code = str(info.get("stock_code") or "").strip()
        resolved_name = str(info.get("company_name") or company_name or "").strip()
        industry = str(info.get("industry") or "").strip()
        listing_date = str(info.get("listing_date") or "").strip()
        website = str(info.get("website") or "").strip()
        profile = str(info.get("profile") or "").strip()

        parts = []
        if industry:
            parts.append(f"所属行业：{industry}")
        if listing_date:
            parts.append(f"上市日期：{listing_date}")
        if website:
            parts.append(f"公司网站：{website}")
        if profile:
            parts.append(f"公司简介：{profile}")

        if not parts:
            return {}

        code_text = f"({stock_code})" if stock_code else ""
        normalized_code = str(info.get("full_code") or stock_code or "").lower()
        lookup_url = (
            f"https://github.com/mouday/stock-open-api?code={normalized_code}"
            if normalized_code
            else "https://github.com/mouday/stock-open-api"
        )

        return {
            "type": "announcement",
            "title": f"{resolved_name}{code_text} 公司基础信息更新",
            "url": lookup_url,
            "source": "stock_open_api",
            "publish_date": date_range[1],
            "content": "；".join(parts),
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "api": "stock-open-api",
                "raw": info.get("raw", {}),
            },
        }

    async def _collect_social_media(
        self,
        query: str,
        date_range: Tuple[str, str],
        strategy: Optional[Dict[str, Any]] = None,
        limit: int = 12,
    ) -> List[Dict[str, Any]]:
        return await self._query_source(
            source_type="social",
            query=query,
            date_range=date_range,
            limit=limit,
            strategy=strategy,
        )

    async def _collect_forums(
        self,
        query: str,
        date_range: Tuple[str, str],
        strategy: Optional[Dict[str, Any]] = None,
        limit: int = 12,
    ) -> List[Dict[str, Any]]:
        return await self._query_source(
            source_type="forum",
            query=query,
            date_range=date_range,
            limit=limit,
            strategy=strategy,
        )

    async def _query_source(
        self,
        source_type: str,
        query: str,
        date_range: Tuple[str, str],
        limit: int,
        strategy: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        strategy = strategy or {}
        domain_filters = (
            strategy.get("domain_filters", {}).get(source_type, {})
            if isinstance(strategy.get("domain_filters"), dict)
            else {}
        )
        include_domains = domain_filters.get("include_domains", []) if isinstance(domain_filters, dict) else []
        exclude_domains = domain_filters.get("exclude_domains", []) if isinstance(domain_filters, dict) else []
        topic = "general" if source_type in {"social", "forum"} else "news"

        payload = {
            "query": query,
            "date_from": date_range[0],
            "date_to": date_range[1],
            "limit": limit,
            "topic": topic,
            "search_depth": "advanced",
        }
        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains

        try:
            results = await news_search.ainvoke(payload)
        except Exception as exc:
            logger.warning("%s source query failed: %s", source_type, exc)
            return []

        if (not isinstance(results, list) or not results) and include_domains:
            # If domain constraint is too strict, retry once with broader coverage.
            relaxed_payload = {
                "query": query,
                "date_from": date_range[0],
                "date_to": date_range[1],
                "limit": limit,
                "topic": topic,
                "search_depth": "advanced",
            }
            if exclude_domains:
                relaxed_payload["exclude_domains"] = exclude_domains
            try:
                results = await news_search.ainvoke(relaxed_payload)
            except Exception as exc:
                logger.warning("%s relaxed source query failed: %s", source_type, exc)
                return []

        return self._normalize_documents_from_tool(
            source_type=source_type,
            results=results,
            default_source=source_type,
        )

    def _normalize_documents_from_tool(
        self,
        source_type: str,
        results: Any,
        default_source: str,
    ) -> List[Dict[str, Any]]:
        if not isinstance(results, list):
            return []

        now_ts = datetime.now().isoformat()
        normalized: List[Dict[str, Any]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "type": source_type,
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "source": item.get("source", default_source),
                    "publish_date": item.get("publish_date", ""),
                    "content": item.get("content", ""),
                    "timestamp": now_ts,
                }
            )
        return normalized

    def _deduplicate(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        unique_docs = []
        for doc in documents:
            url = (doc.get("url") or "").strip()
            key_seed = url or "|".join(
                [
                    str(doc.get("type", "")),
                    str(doc.get("title", "")),
                    str(doc.get("publish_date", "")),
                    str(doc.get("content", ""))[:150],
                ]
            )
            key = hashlib.md5(key_seed.encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            unique_docs.append(doc)
        return unique_docs

    async def _enrich_documents(
        self,
        documents: List[Dict[str, Any]],
        company_name: str,
        stock_code: Optional[str],
        aliases: List[str],
        entity_basis: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        entity_basis = entity_basis or []
        enriched: List[Dict[str, Any]] = []
        for doc in documents:
            normalized = self._normalize_document(
                doc=doc,
                company_name=company_name,
                stock_code=stock_code,
                aliases=aliases,
            )
            content = normalized.get("content", "")
            if content:
                try:
                    entities = await entity_recognition.ainvoke({"text": content})
                except Exception as exc:
                    logger.warning("entity pre-recognition failed: %s", exc)
                    entities = []
                normalized["entities_hint"] = self._merge_entities_with_basis(
                    extracted_entities=entities if isinstance(entities, list) else [],
                    entity_basis=entity_basis,
                    text_blob=f"{normalized.get('title', '')} {content}",
                )
            enriched.append(normalized)
        return enriched

    def _merge_entities_with_basis(
        self,
        extracted_entities: List[Dict[str, Any]],
        entity_basis: List[Dict[str, Any]],
        text_blob: str,
    ) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen = set()

        for entity in extracted_entities:
            if not isinstance(entity, dict):
                continue
            text = str(entity.get("text") or "").strip()
            etype = str(entity.get("type") or "").strip() or "UNKNOWN"
            if not text:
                continue
            key = (text, etype)
            if key in seen:
                continue
            seen.add(key)
            merged.append(entity)

        for base in entity_basis:
            if not isinstance(base, dict):
                continue
            text = str(base.get("text") or "").strip()
            etype = str(base.get("type") or "").strip() or "UNKNOWN"
            if not text:
                continue

            # Canonical entity and stock code are always included; aliases require mention.
            mentioned = text in text_blob
            if etype not in {"COMPANY", "STOCK_CODE"} and not mentioned:
                continue

            key = (text, etype)
            if key in seen:
                continue
            seen.add(key)
            merged.append(
                {
                    **base,
                    "confidence": max(float(base.get("confidence", 0.0)), 0.95),
                    "from_entity_basis": True,
                    "mentioned": mentioned,
                }
            )

        return merged

    def _normalize_document(
        self,
        doc: Dict[str, Any],
        company_name: str,
        stock_code: Optional[str],
        aliases: List[str],
    ) -> Dict[str, Any]:
        raw_title = doc.get("title", "")
        raw_content = doc.get("content", "")
        metadata = dict(doc.get("metadata") or {})

        title = self._clean_text(raw_title)
        unreadable_reason = str(
            metadata.get("crawl_unreadable_reason")
            or metadata.get("unreadable_reason")
            or self._detect_unreadable_reason(raw_content)
            or ""
        ).strip()
        content = "" if unreadable_reason else self._clean_text(raw_content)
        if not content and not unreadable_reason:
            content = title
        publish_date = self._to_date_str(doc.get("publish_date") or datetime.now())
        source = (doc.get("source") or doc.get("type") or "unknown").strip()
        url = (doc.get("url") or "").strip()

        relevance = self._score_relevance(
            title=title,
            content=content,
            company_name=company_name,
            stock_code=stock_code,
            aliases=aliases,
        )
        if unreadable_reason:
            relevance = min(relevance, 0.35)

        metadata.update(
            {
                "cleaned": True,
                "company_name": company_name,
                "stock_code": stock_code,
                "aliases": aliases,
                "unreadable_content": bool(unreadable_reason),
                "unreadable_reason": unreadable_reason,
            }
        )

        return {
            **doc,
            "type": doc.get("type", "news"),
            "title": title or "未命名舆情",
            "content": content,
            "publish_date": publish_date,
            "source": source,
            "url": url,
            "company_name": company_name,
            "stock_code": stock_code,
            "relevance_score": relevance,
            "metadata": metadata,
        }

    def _detect_unreadable_reason(self, text: Any) -> str:
        raw = str(text or "")
        if not raw:
            return ""

        sample = raw[:12000]
        lowered = sample.lower()

        pdf_markers = [
            "%pdf",
            "endobj",
            "stream",
            "endstream",
            "xref",
            "trailer",
            "/type /page",
            "/type/page",
            "/length",
            "/filter",
        ]
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

    def _clean_text(self, text: Any) -> str:
        raw = str(text or "")
        if not raw:
            return ""
        clean = re.sub(r"<[^>]+>", " ", raw)
        clean = re.sub(r"https?://\S+", " ", clean)
        clean = re.sub(r"&[a-zA-Z]+;", " ", clean)
        clean = re.sub(r"[\r\n\t]+", " ", clean)
        clean = re.sub(r"\s{2,}", " ", clean)
        return clean.strip()

    def _score_relevance(
        self,
        title: str,
        content: str,
        company_name: str,
        stock_code: Optional[str],
        aliases: List[str],
    ) -> float:
        text_blob = f"{title} {content}"
        all_terms = self._prepare_company_terms(company_name, stock_code, aliases)
        title_hits, content_hits, total_hits = self._count_company_term_hits(title, content, all_terms)
        primary_terms = [token for token in [company_name, stock_code] if token]
        _, _, primary_hits = self._count_company_term_hits(title, content, primary_terms)

        base = 0.15
        if primary_hits >= 1:
            base += 0.35
        if title_hits >= 1:
            base += 0.25
        if content_hits >= 1:
            base += 0.15
        base += min(max(0, total_hits - 1) * 0.08, 0.24)
        if any(k in text_blob for k in ["公告", "处罚", "调查", "诉讼", "业绩", "合作"]):
            base += 0.1
        return round(min(1.0, max(0.0, base)), 2)

    def _build_fallback_documents(
        self,
        company_name: str,
        stock_code: Optional[str],
        date_range: Tuple[str, str],
        sources: List[str],
    ) -> List[Dict[str, Any]]:
        start_date, end_date = date_range
        code_text = f"({stock_code})" if stock_code else ""

        templates = [
            {
                "type": "announcement",
                "title": f"{company_name}{code_text} 经营情况说明公告",
                "content": (
                    f"{company_name}{code_text} 披露经营情况更新，近期主营业务保持稳定。"
                    "公告同时提示行业竞争与成本波动风险。"
                ),
            },
            {
                "type": "news",
                "title": f"{company_name} 舆情跟踪：市场关注业绩与监管动态",
                "content": (
                    f"媒体报道显示 {company_name} 讨论热度上升，市场重点关注业绩变化。"
                    "同时关注是否存在处罚、调查、诉讼等潜在风险事件。"
                ),
            },
            {
                "type": "forum",
                "title": f"{company_name} 投资者讨论摘要",
                "content": (
                    f"投资者围绕 {company_name} 后续走势展开讨论，观点存在分歧。"
                    "一部分看好合作和订单进展，另一部分担忧盈利能力下滑。"
                ),
            },
        ]

        allowed = set(sources or [])
        docs: List[Dict[str, Any]] = []
        for idx, tpl in enumerate(templates):
            if allowed and tpl["type"] not in allowed:
                continue
            docs.append(
                {
                    "type": tpl["type"],
                    "title": tpl["title"],
                    "url": f"local://fallback/{company_name}/{idx}",
                    "source": "local_fallback",
                    "publish_date": end_date if idx == 0 else start_date,
                    "content": tpl["content"],
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {"is_fallback": True},
                }
            )
        return docs


def create_data_collection_agent(
    llm: BaseLanguageModel,
    tools: Optional[List[BaseTool]] = None,
) -> Any:
    """
    Create data collection agent.
    """
    return DataCollectionAgent(llm=llm, tools=tools)



