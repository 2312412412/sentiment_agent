"""
Tools for search, crawling, sentiment analysis, entity recognition, and event extraction.
"""
import inspect
import logging
import hashlib
import re
import asyncio
from dataclasses import asdict, is_dataclass
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
from langchain_core.tools import BaseTool, tool
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from dotenv import load_dotenv
from pydantic import BaseModel, Field

try:
    from langchain_tavily import TavilySearch
except ImportError:  # pragma: no cover - optional dependency
    TavilySearch = None

logger = logging.getLogger(__name__)


def _load_project_env() -> None:
    """Load .env from project root so tools work even without importing config first."""
    try:
        env_path = Path(__file__).resolve().parents[1] / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            return
        load_dotenv()
    except Exception:
        # Do not block tool initialization if dotenv loading fails.
        return


_load_project_env()


POSITIVE_WORDS = [
    "利好",
    "增长",
    "上涨",
    "优秀",
    "突破",
    "成功",
    "盈利",
    "预喜",
    "回购",
    "增持",
    "创新高",
    "改善",
    "超预期",
    "签约",
    "合作",
    "中标",
]
NEGATIVE_WORDS = [
    "利空",
    "下跌",
    "亏损",
    "违规",
    "处罚",
    "风险",
    "立案",
    "调查",
    "诉讼",
    "下滑",
    "减持",
    "暴跌",
    "违约",
    "停牌",
    "质押",
    "问询",
    "冻结",
]
COMPANY_SUFFIXES = [
    "集团股份有限公司",
    "股份有限公司",
    "集团有限公司",
    "控股有限公司",
    "科技有限公司",
    "有限责任公司",
    "有限公司",
    "集团",
    "公司",
]
LEADING_NOISE_TOKENS = [
    "并与",
    "以及",
    "关于",
    "根据",
    "对于",
    "公司",
    "公告",
    "披露",
    "发布",
    "称为",
    "表示",
    "中",
    "和",
    "对",
    "在",
    "将",
]
EVENT_PATTERNS = {
    "融资": ["融资", "定增", "配股", "发债", "募资", "增发"],
    "并购": ["并购", "收购", "重组", "兼并", "整合"],
    "处罚": ["处罚", "罚款", "立案", "调查", "违规", "问询函"],
    "业绩": ["业绩", "盈利", "亏损", "净利润", "营收", "预增", "预亏"],
    "人事": ["辞职", "任免", "任命", "高管", "董事长", "总经理"],
    "诉讼": ["诉讼", "起诉", "仲裁", "判决", "纠纷"],
    "合作": ["合作", "签约", "协议", "战略合作", "合资", "中标"],
}
EVENT_TITLE_LABELS = {
    "融资": "financing",
    "并购": "merger",
    "处罚": "penalty",
    "业绩": "earnings",
    "人事": "personnel",
    "诉讼": "litigation",
    "合作": "cooperation",
}
ENTITY_STRIP_CHARS = " \t\r\n,.;:!?，。；：、（）()【】[]《》<>\"'“”‘’"


def _clip_text(text: str, limit: int = 200) -> str:
    return text if len(text) <= limit else f"{text[:limit]}..."


def _dedupe_dict_items(items: List[Dict[str, Any]], keys: List[str]) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for item in items:
        key = tuple(item.get(k) for k in keys)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _extract_sentence(text: str, trigger: str, window: int = 4000) -> str:
    pos = text.find(trigger)
    if pos == -1:
        return ""
    start = max(0, pos - window)
    end = min(len(text), pos + len(trigger) + window)
    snippet = text[start:end].strip()
    return re.sub(r"\s+", " ", snippet)


def _clean_entity_text(text: str) -> str:
    cleaned = str(text or "").strip(ENTITY_STRIP_CHARS)
    changed = True
    while changed and cleaned:
        changed = False
        for token in LEADING_NOISE_TOKENS:
            if cleaned.startswith(token):
                cleaned = cleaned[len(token):].strip(ENTITY_STRIP_CHARS)
                changed = True
                break
    return cleaned


def _parse_iso_date(value: Optional[str]) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def _infer_tavily_time_range(
    date_from: Optional[str],
    date_to: Optional[str],
) -> Optional[str]:
    start_dt = _parse_iso_date(date_from)
    end_dt = _parse_iso_date(date_to)
    if not start_dt and not end_dt:
        return None

    today = datetime.now().date()
    end_date = end_dt.date() if end_dt else today
    start_date = start_dt.date() if start_dt else end_date
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    span_days = max(1, (end_date - start_date).days + 1)
    end_lag_days = max(0, (today - end_date).days)

    if span_days <= 1 and end_lag_days <= 1:
        return "day"
    if span_days <= 7 and end_lag_days <= 7:
        return "week"
    if span_days <= 31 and end_lag_days <= 31:
        return "month"
    return "year"


def _extract_pdf_text(payload: bytes, max_chars: int) -> str:
    if not payload:
        return ""

    try:
        from io import BytesIO
        from pypdf import PdfReader
    except Exception:
        return ""

    try:
        reader = PdfReader(BytesIO(payload))
        chunks: List[str] = []
        current = 0
        for page in reader.pages[:100]:
            page_text = re.sub(r"\s+", " ", str(page.extract_text() or "")).strip()
            if not page_text:
                continue
            remaining = max_chars - current
            if remaining <= 0:
                break
            if len(page_text) > remaining:
                page_text = page_text[:remaining]
            chunks.append(page_text)
            current += len(page_text) + 1
            if current >= max_chars:
                break
        return " ".join(chunks).strip()
    except Exception as exc:
        logger.debug("PDF text extraction failed: %s", exc)
        return ""


# ========== Search tools ==========

@tool
async def news_search(
    query: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 10,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    topic: str = "news",
    search_depth: str = "advanced",
) -> List[Dict[str, Any]]:
    """Search news-like documents with Tavily."""
    logger.info("Running Tavily search: %s", query)

    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if TavilySearch is None:
        logger.warning("langchain-tavily is not installed; returning empty search results")
        return []
    if not api_key:
        logger.warning("TAVILY_API_KEY is missing; returning empty search results")
        return []

    include_domains = [
        str(domain or "").strip().lower()
        for domain in (include_domains or [])
        if str(domain or "").strip()
    ]
    include_domains = list(dict.fromkeys(include_domains))
    exclude_domains = [
        str(domain or "").strip().lower()
        for domain in (exclude_domains or [])
        if str(domain or "").strip()
    ]
    exclude_domains = list(dict.fromkeys(exclude_domains))
    normalized_topic = topic if topic in {"news", "general", "finance"} else "news"
    normalized_depth = search_depth if search_depth in {"basic", "advanced"} else "advanced"

    try:
        tavily = TavilySearch(
            max_results=max(1, min(limit, 20)),
            topic=normalized_topic,
            include_answer=False,
            include_raw_content=False,
            search_depth=normalized_depth,
            include_domains=include_domains or None,
            exclude_domains=exclude_domains or None,
        )

        invoke_args: Dict[str, Any] = {"query": query}
        inferred_time_range = _infer_tavily_time_range(date_from, date_to)
        if inferred_time_range:
            invoke_args["time_range"] = inferred_time_range
        if include_domains:
            invoke_args["include_domains"] = include_domains
        if exclude_domains:
            invoke_args["exclude_domains"] = exclude_domains

        try:
            payload = await tavily.ainvoke(invoke_args)
        except Exception:
            payload = tavily.invoke(invoke_args)

    except Exception as exc:
        logger.error("Tavily search failed: %s", exc)
        return []

    results: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        raw = payload.get("results", [])
        if isinstance(raw, list):
            results = [x for x in raw if isinstance(x, dict)]
    elif isinstance(payload, list):
        results = [x for x in payload if isinstance(x, dict)]

    normalized: List[Dict[str, Any]] = []
    for item in results[:limit]:
        url = str(item.get("url", "")).strip()
        host = "tavily"
        if url:
            try:
                host = urlparse(url).netloc.lower().replace("www.", "") or "tavily"
            except Exception:
                host = "tavily"

        normalized.append({
            "title": str(item.get("title", "")).strip(),
            "url": url,
            "source": host,
            "publish_date": str(
                item.get("published_date")
                or item.get("published_time")
                or item.get("date")
                or ""
            ),
            "content": str(
                item.get("content")
                or item.get("raw_content")
                or item.get("snippet")
                or ""
            ).strip(),
        })
    return normalized


@tool
async def web_crawl(
    url: str,
    selector: Optional[str] = None,
    max_chars: int = 6000,
    include_links: bool = False,
    timeout: int = 20,
) -> Dict[str, Any]:
    """
    通用网页爬取工具，抓取指定 URL 的正文文本（可选 CSS 选择器）。
    """
    logger.info("web_crawl url=%s selector=%s", url, selector or "")

    target_url = str(url or "").strip()
    parsed = urlparse(target_url)
    if not target_url or parsed.scheme not in {"http", "https"}:
        return {"url": target_url, "error": "invalid_url"}

    max_chars = max(200, min(max_chars, 20000))
    timeout = max(3, min(timeout, 60))

    def _request() -> requests.Response:
        return requests.get(
            target_url,
            timeout=timeout,
            allow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0.0.0 Safari/537.36 sentiment-agent/1.0"
                )
            },
        )

    try:
        resp = await asyncio.to_thread(_request)
    except Exception as exc:
        logger.warning("web_crawl request failed for %s: %r", target_url, exc)
        return {"url": target_url, "error": "request_failed", "detail": repr(exc)}

    content_type = str(resp.headers.get("Content-Type", "")).lower()
    if not resp.encoding:
        try:
            resp.encoding = resp.apparent_encoding
        except Exception:
            pass

    raw_bytes = resp.content or b""
    is_pdf_response = "application/pdf" in content_type or raw_bytes.startswith(b"%PDF")

    if is_pdf_response:
        pdf_text = _extract_pdf_text(raw_bytes, max_chars=max_chars)
        if pdf_text:
            return {
                "url": target_url,
                "final_url": str(resp.url or target_url),
                "status_code": int(resp.status_code),
                "source": urlparse(str(resp.url or target_url)).netloc.lower().replace("www.", ""),
                "content_type": content_type or "application/pdf",
                "title": "",
                "content": _clip_text(pdf_text, max_chars),
                "fetched_at": datetime.now().isoformat(),
            }

        return {
            "url": target_url,
            "final_url": str(resp.url or target_url),
            "status_code": int(resp.status_code),
            "source": urlparse(str(resp.url or target_url)).netloc.lower().replace("www.", ""),
            "content_type": content_type or "application/pdf",
            "title": "",
            "content": "",
            "fetched_at": datetime.now().isoformat(),
            "unreadable": True,
            "unreadable_reason": "pdf_unreadable_or_scanned",
        }

    title = ""
    text_content = ""
    links: List[str] = []

    html_hint = ""
    try:
        html_hint = (resp.text or "")[:2048].lower()
    except Exception:
        html_hint = ""

    if "html" in content_type or "<html" in html_hint:
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()

        title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
        root = soup

        if selector:
            selected = soup.select(selector)
            if selected:
                text_content = " ".join(
                    re.sub(r"\s+", " ", node.get_text(" ", strip=True))
                    for node in selected
                ).strip()
                if include_links:
                    seen = set()
                    for node in selected:
                        for a in node.select("a[href]"):
                            href = str(a.get("href") or "").strip()
                            if not href:
                                continue
                            absolute = urljoin(resp.url, href)
                            if absolute in seen:
                                continue
                            seen.add(absolute)
                            links.append(absolute)

        if not text_content:
            body = soup.body if soup.body else soup
            text_content = re.sub(r"\s+", " ", body.get_text(" ", strip=True)).strip()

        if include_links and not links:
            seen = set()
            for a in root.select("a[href]"):
                href = str(a.get("href") or "").strip()
                if not href:
                    continue
                absolute = urljoin(resp.url, href)
                if absolute in seen:
                    continue
                seen.add(absolute)
                links.append(absolute)
    else:
        if any(
            token in content_type
            for token in ["text/", "json", "xml", "javascript"]
        ) or not content_type:
            text_content = re.sub(r"\s+", " ", str(resp.text or "")).strip()
        else:
            return {
                "url": target_url,
                "final_url": str(resp.url or target_url),
                "status_code": int(resp.status_code),
                "source": urlparse(str(resp.url or target_url)).netloc.lower().replace("www.", ""),
                "content_type": content_type or "",
                "title": "",
                "content": "",
                "fetched_at": datetime.now().isoformat(),
                "unreadable": True,
                "unreadable_reason": "unsupported_binary_content_type",
            }

    result: Dict[str, Any] = {
        "url": target_url,
        "final_url": str(resp.url or target_url),
        "status_code": int(resp.status_code),
        "source": urlparse(str(resp.url or target_url)).netloc.lower().replace("www.", ""),
        "content_type": content_type or "",
        "title": title,
        "content": _clip_text(text_content, max_chars),
        "fetched_at": datetime.now().isoformat(),
    }
    if include_links:
        result["links"] = links[:100]
    return result


def _normalize_stock_code(stock_code: str, market: str = "auto") -> str:
    raw = str(stock_code or "").strip()
    if not raw:
        return ""

    if re.fullmatch(r"[A-Za-z]{2}\d{6}", raw):
        return raw.lower()

    digits = re.sub(r"\D", "", raw)
    if not re.fullmatch(r"\d{6}", digits):
        return raw.lower()

    market_lower = (market or "auto").strip().lower()
    if market_lower in {"sh", "sz", "bj"}:
        return f"{market_lower}{digits}"

    if digits[0] in {"5", "6", "9"}:
        prefix = "sh"
    elif digits[0] in {"0", "1", "2", "3"}:
        prefix = "sz"
    elif digits[0] in {"4", "8"}:
        prefix = "bj"
    else:
        prefix = "sh"
    return f"{prefix}{digits}"


def _stock_open_lookup_sync(code: str) -> Dict[str, Any]:
    from stock_open_api.api.eastmoney import company as eastmoney_company

    return eastmoney_company.get_company_info(code)


def _normalize_stock_company_payload(
    payload: Dict[str, Any],
    requested_code: str,
) -> Dict[str, Any]:
    stock_code = str(
        payload.get("A股代码")
        or payload.get("stock_code")
        or re.sub(r"\D", "", requested_code)[:6]
    ).strip()
    company_name = str(
        payload.get("公司名称")
        or payload.get("A股简称")
        or payload.get("company_name")
        or ""
    ).strip()
    listing_date = str(
        payload.get("上市日期")
        or payload.get("listing_date")
        or ""
    ).strip()
    industry = str(
        payload.get("所属东财行业")
        or payload.get("所属证监会行业")
        or payload.get("industry")
        or ""
    ).strip()
    website = str(payload.get("公司网站") or payload.get("website") or "").strip()
    profile = str(payload.get("公司介绍") or payload.get("profile") or "").strip()

    return {
        "stock_code": stock_code,
        "full_code": requested_code.lower(),
        "company_name": company_name,
        "listing_date": listing_date,
        "industry": industry,
        "website": website,
        "profile": profile,
        "raw": payload,
    }


# ========== Stock tools ==========

@tool
async def stock_open_company_info(
    stock_code: str,
    market: str = "auto",
) -> Dict[str, Any]:
    """Fetch listed-company profile data through stock-open-api."""
    normalized = _normalize_stock_code(stock_code=stock_code, market=market)
    if not normalized:
        return {}

    try:
        import stock_open_api  # noqa: F401
    except ImportError:
        logger.warning("stock-open-api is not installed; company profile lookup is unavailable")
        return {}

    candidates = [normalized]
    digits = re.sub(r"\D", "", normalized)
    if (market or "auto").strip().lower() == "auto" and re.fullmatch(r"\d{6}", digits):
        for prefix in ("sh", "sz", "bj"):
            candidate = f"{prefix}{digits}"
            if candidate not in candidates:
                candidates.append(candidate)

    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            payload = await asyncio.to_thread(_stock_open_lookup_sync, candidate)
            if isinstance(payload, dict) and payload:
                return _normalize_stock_company_payload(payload, requested_code=candidate)
        except Exception as exc:
            last_error = exc
            continue

    logger.warning("stock-open-api query failed for %s: %s", stock_code, last_error)
    return {}


# ========== Analysis tools ==========

@tool
async def sentiment_analysis(text: str) -> Dict[str, Any]:
    """Analyze sentiment polarity for Chinese financial text."""
    logger.info("Running sentiment analysis: %s", _clip_text(text, 50))

    clean_text = re.sub(r"\s+", "", text)
    if not clean_text:
        return {"label": "neutral", "score": 0.5, "aspects": {}, "evidence": []}

    positive_hits = [word for word in POSITIVE_WORDS if word in clean_text]
    negative_hits = [word for word in NEGATIVE_WORDS if word in clean_text]

    pos_count = len(positive_hits)
    neg_count = len(negative_hits)
    total = pos_count + neg_count

    if pos_count > neg_count:
        label = "positive"
        score = min(0.55 + pos_count * 0.08 - neg_count * 0.03, 0.95)
    elif neg_count > pos_count:
        label = "negative"
        score = min(0.55 + neg_count * 0.08 - pos_count * 0.03, 0.95)
    else:
        label = "neutral"
        score = 0.5 if total == 0 else min(0.5 + total * 0.03, 0.65)

    aspects = {
        "profitability": round(
            min(1.0, sum(w in clean_text for w in ["盈利", "亏损", "净利润", "营收"]) * 0.3),
            2,
        ),
        "compliance": round(
            min(1.0, sum(w in clean_text for w in ["违规", "处罚", "调查", "诉讼"]) * 0.35),
            2,
        ),
        "market": round(
            min(1.0, sum(w in clean_text for w in ["上涨", "下跌", "利好", "利空"]) * 0.25),
            2,
        ),
    }

    evidence = positive_hits[:3] + negative_hits[:3]
    return {
        "label": label,
        "score": round(max(0.0, min(score, 1.0)), 2),
        "aspects": aspects,
        "evidence": evidence,
    }


# ========== Entity tools ==========

@tool
async def entity_recognition(text: str) -> List[Dict[str, Any]]:
    """Recognize company, person, organization, and stock-code entities."""
    logger.info("Running entity recognition: %s...", text[:50])

    entities: List[Dict[str, Any]] = []

    company_pattern = re.compile(
        r"([\u4e00-\u9fffA-Za-z]{2,32}(?:集团股份有限公司|股份有限公司|集团有限公司|控股有限公司|科技有限公司|有限责任公司|有限公司|集团))"
    )
    person_with_role_pattern = re.compile(
        r"(?<!有限公司)(?<!股份有限公司)([\u4e00-\u9fff]{2,4})(董事长|总经理|董事|监事|高管|董秘|总裁)"
    )
    role_then_person_pattern = re.compile(
        r"(董事长|总经理|董事|监事|高管|董秘|总裁)([\u4e00-\u9fff]{2,4})(?=表示|称|指出|认为|介绍|[，,\s])"
    )
    stock_code_pattern = re.compile(r"(?<!\d)(60\d{4}|68\d{4}|00\d{4}|30\d{4})(?!\d)")
    org_pattern = re.compile(r"([\u4e00-\u9fff]{2,20}(?:证监会|交易所|法院|银行|证券|基金))")

    for match in company_pattern.finditer(text):
        company = _clean_entity_text(match.group(1))
        if not company:
            continue
        entities.append({
            "text": company,
            "type": "COMPANY",
            "confidence": 0.9,
            "start": match.start(),
            "end": match.end(),
        })

        for suffix in COMPANY_SUFFIXES:
            if company.endswith(suffix):
                short_name = company[: -len(suffix)].strip()
                if 2 <= len(short_name) <= 12:
                    entities.append({
                        "text": short_name,
                        "type": "COMPANY_ALIAS",
                        "canonical_name": company,
                        "confidence": 0.72,
                    })
                break

    for match in person_with_role_pattern.finditer(text):
        person_name = _clean_entity_text(match.group(1))
        if not person_name or "公司" in person_name:
            continue
        entities.append({
            "text": person_name,
            "type": "PERSON",
            "role": match.group(2),
            "confidence": 0.78,
            "start": match.start(1),
            "end": match.end(2),
        })

    for match in role_then_person_pattern.finditer(text):
        person_name = _clean_entity_text(match.group(2))
        if not person_name or "公司" in person_name:
            continue
        entities.append({
            "text": person_name,
            "type": "PERSON",
            "role": match.group(1),
            "confidence": 0.82,
            "start": match.start(2),
            "end": match.end(2),
        })

    for match in stock_code_pattern.finditer(text):
        entities.append({
            "text": match.group(1),
            "type": "STOCK_CODE",
            "confidence": 0.98,
            "start": match.start(),
            "end": match.end(),
        })

    for match in org_pattern.finditer(text):
        org = _clean_entity_text(match.group(1))
        if not org:
            continue
        entities.append({
            "text": org,
            "type": "ORG",
            "confidence": 0.75,
            "start": match.start(),
            "end": match.end(),
        })

    return _dedupe_dict_items(entities, ["text", "type"])


# ========== Event tools ==========

@tool
async def event_extraction(
    text: str,
    entities: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Extract coarse-grained event signals from text."""
    logger.info("Running event extraction: %s...", text[:50])

    mentioned_entities = entities or []
    events: List[Dict[str, Any]] = []

    for event_type, triggers in EVENT_PATTERNS.items():
        for trigger in triggers:
            if trigger not in text:
                continue

            events.append({
                "id": hashlib.md5(f"{event_type}:{trigger}:{text[:80]}".encode("utf-8")).hexdigest()[:12],
                "type": event_type,
                "title": f"{EVENT_TITLE_LABELS.get(event_type, event_type)} event",
                "trigger": trigger,
                "arguments": {
                    "entities": mentioned_entities[:5],
                },
                "impact": _extract_sentence(text, trigger),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "confidence": 0.82,
            })

    return _dedupe_dict_items(events, ["type", "trigger"])


# ========== Agent tools ==========

class AgentCaller:
    """Async helper for calling other agents from the tool layer."""

    def __init__(self, agents_registry: Dict[str, Any]):
        """Initialize the agent registry."""
        self.agents_registry = agents_registry

    async def call(
        self,
        agent_name: str,
        task: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Call a registered agent with a compatible execute signature."""
        if agent_name not in self.agents_registry:
            raise ValueError(f"Unknown agent: {agent_name}")

        agent = self.agents_registry[agent_name]
        logger.info("Calling agent=%s task=%s", agent_name, task)

        context = dict(context or {})
        execute = getattr(agent, "execute", None)
        if execute is None:
            raise TypeError(f"Agent {agent_name} does not define execute()")

        parameters = inspect.signature(execute).parameters
        if "task" in parameters:
            return await agent.execute(task=task, **context)
        if context:
            return await agent.execute(**context)
        raise TypeError(
            f"Agent {agent_name} execute() does not accept a `task` argument; "
            "pass explicit context arguments instead."
        )

    async def call_parallel(
        self,
        calls: List[Dict[str, Any]],
    ) -> List[Any]:
        """Call multiple agents in parallel."""
        tasks = [
            self.call(call["agent_name"], call.get("task", ""), call.get("context"))
            for call in calls
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results


class AgentCallerToolInput(BaseModel):
    """Schema for the `agent_caller` tool."""

    agent_name: str = Field(
        ...,
        description="Name of the earlier agent to call.",
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments forwarded to the target agent's execute() method.",
    )
    task: str = Field(
        default="",
        description="Optional task string for agents whose execute() accepts a task parameter.",
    )


def _make_json_safe(value: Any) -> Any:
    """Convert agent outputs into JSON-safe primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _make_json_safe(value.to_dict())
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _make_json_safe(value.model_dump())
    if is_dataclass(value):
        return _make_json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(item) for item in value]
    return str(value)


def create_agent_caller_tool(
    agents_registry: Dict[str, Any],
    allowed_agents: Optional[List[str]] = None,
) -> BaseTool:
    """Build a tool that can call earlier agents for supplemental context."""
    caller = AgentCaller(agents_registry=agents_registry)
    allowed = [
        name
        for name in (allowed_agents or list(agents_registry.keys()))
        if name in agents_registry
    ]
    allowed_set = set(allowed)
    allowed_text = ", ".join(allowed) if allowed else "none"

    async def _call_previous_agent(
        agent_name: str,
        context: Optional[Dict[str, Any]] = None,
        task: str = "",
    ) -> Dict[str, Any]:
        if agent_name not in allowed_set:
            raise ValueError(
                f"agent_caller only allows previous agents: {allowed_text}. "
                f"Received: {agent_name}"
            )

        result = await caller.call(
            agent_name=agent_name,
            task=task,
            context=dict(context or {}),
        )
        return {
            "agent_name": agent_name,
            "task": task,
            "result": _make_json_safe(result),
        }

    return tool(
        "agent_caller",
        args_schema=AgentCallerToolInput,
        description=(
            "Call an earlier pipeline agent when the current step needs supplemental output. "
            f"Allowed agents: {allowed_text}. "
            "Pass execute() keyword arguments through `context`. "
            "Only use `task` when the target agent explicitly supports it."
        ),
    )(_call_previous_agent)


# ========== Tool registry ==========

def get_default_tools() -> List[BaseTool]:
    """Return the default tool list."""
    return [
        news_search,
        web_crawl,
        stock_open_company_info,
        sentiment_analysis,
        entity_recognition,
        event_extraction,
    ]
