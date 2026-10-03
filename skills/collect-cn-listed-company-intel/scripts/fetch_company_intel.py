#!/usr/bin/env python3
"""Collect source-linked intelligence for Chinese listed companies."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib import error, parse, request

DEFAULT_SOURCES = ("tavily", "weibo", "cninfo", "sse")
DEFAULT_USER_AGENT = "codex-collect-cn-listed-company-intel/1.0"
WEIBO_MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


class SourceError(RuntimeError):
    """Raised when a source request cannot be completed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect company intelligence from Tavily, Weibo, CNINFO, and SSE.",
    )
    parser.add_argument("--company-name", help="Human-readable company name.")
    parser.add_argument("--stock-code", help="Listed company stock code, e.g. 600519 or 300750.")
    parser.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCES),
        help="Comma-separated sources: tavily,weibo,cninfo,sse",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Rolling lookback window in days. Default: 30",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=10,
        help="Maximum normalized items to keep per source. Default: 10",
    )
    parser.add_argument("--output", help="Write JSON output to this path instead of stdout.")
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=20,
        help="Network timeout in seconds. Default: 20",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help=f"HTTP User-Agent header. Default: {DEFAULT_USER_AGENT}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Emit planned requests without making network calls.",
    )
    parser.add_argument(
        "--fail-on-source-error",
        action="store_true",
        help="Exit non-zero if any requested source fails.",
    )

    parser.add_argument(
        "--tavily-query",
        help="Override the Tavily query. Default builds from company-name and stock-code.",
    )
    parser.add_argument(
        "--tavily-topic",
        default="finance",
        choices=("general", "news", "finance"),
        help="Tavily topic. Default: finance",
    )
    parser.add_argument(
        "--tavily-search-depth",
        default="advanced",
        choices=("basic", "advanced"),
        help="Tavily search depth. Default: advanced",
    )
    parser.add_argument(
        "--tavily-include-domains",
        help="Comma-separated domains to include for Tavily.",
    )
    parser.add_argument(
        "--tavily-exclude-domains",
        help="Comma-separated domains to exclude for Tavily.",
    )

    parser.add_argument("--weibo-uid", help="Weibo user id for the company account.")
    parser.add_argument("--weibo-screen-name", help="Weibo screen name for the company account.")
    parser.add_argument(
        "--weibo-code",
        help="Optional OAuth authorization code. Used only when access token is absent.",
    )

    parser.add_argument(
        "--cninfo-stock",
        help="Exact CNINFO stock token if already known, e.g. 000001,gssz0000001.",
    )
    parser.add_argument(
        "--cninfo-column",
        default="szse",
        help="CNINFO column parameter. Default: szse",
    )
    parser.add_argument(
        "--cninfo-category",
        default="",
        help="CNINFO category filter. Default: empty for all categories.",
    )

    parser.add_argument(
        "--sse-report-type",
        default="ALL",
        help="SSE reportType parameter. Default: ALL",
    )
    parser.add_argument(
        "--sse-report-type2",
        default="",
        help="SSE reportType2 parameter. Default: empty for all secondary types.",
    )

    args = parser.parse_args()
    if args.days < 1:
        parser.error("--days must be >= 1")
    if args.max_items < 1:
        parser.error("--max-items must be >= 1")
    if not args.company_name and not args.stock_code:
        parser.error("Provide at least --company-name or --stock-code.")
    return args


def parse_sources(raw_sources: str) -> list[str]:
    parsed: list[str] = []
    seen: set[str] = set()
    for item in raw_sources.split(","):
        source = item.strip().lower()
        if not source:
            continue
        if source not in DEFAULT_SOURCES:
            raise SourceError(
                f"Unsupported source '{source}'. Choose from {', '.join(DEFAULT_SOURCES)}."
            )
        if source not in seen:
            parsed.append(source)
            seen.add(source)
    if not parsed:
        raise SourceError("No valid sources selected.")
    return parsed


def split_csv(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def compute_window(days: int) -> tuple[str, str]:
    end_date = date.today()
    start_date = end_date - timedelta(days=days - 1)
    return start_date.isoformat(), end_date.isoformat()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_output_dir(path: Path) -> None:
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def mask_secret(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 6:
        return "***"
    return value[:3] + "***" + value[-3:]


def sanitize_headers(headers: dict[str, str]) -> dict[str, str]:
    sanitized = dict(headers)
    authorization = sanitized.get("Authorization")
    if authorization:
        sanitized["Authorization"] = "Bearer " + mask_secret(authorization.replace("Bearer ", ""))
    return sanitized


def sanitize_mapping(values: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in values.items():
        if "token" in key or "secret" in key or key in {"code", "client_id"}:
            sanitized[key] = mask_secret(str(value))
        else:
            sanitized[key] = value
    return sanitized


def build_plan(
    *,
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    query: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    form_body: dict[str, Any] | None = None,
    auth_env: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "method": method,
        "url": url,
        "headers": sanitize_headers(headers or {}),
        "query": sanitize_mapping(query or {}),
        "json_body": sanitize_mapping(json_body or {}),
        "form_body": sanitize_mapping(form_body or {}),
        "auth_env": auth_env or [],
    }


def http_request(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    timeout: int,
    query: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    form_body: dict[str, Any] | None = None,
) -> str:
    full_url = url
    if query:
        query_string = parse.urlencode(query, doseq=True)
        separator = "&" if "?" in full_url else "?"
        full_url = f"{full_url}{separator}{query_string}"

    payload: bytes | None = None
    request_headers = dict(headers)
    if json_body is not None:
        payload = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    elif form_body is not None:
        payload = parse.urlencode(form_body).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/x-www-form-urlencoded")

    req = request.Request(full_url, data=payload, headers=request_headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SourceError(f"HTTP {exc.code} from {url}: {body[:400]}") from exc
    except error.URLError as exc:
        raise SourceError(f"Network error for {url}: {exc.reason}") from exc


def load_json_payload(payload: str) -> dict[str, Any]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        match = re.search(r"^[^(]*\((.*)\)\s*;?\s*$", payload, re.S)
        if not match:
            raise SourceError("Response was not valid JSON or JSONP.")
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise SourceError("JSONP wrapper found, but payload JSON was invalid.") from exc
    if not isinstance(data, dict):
        raise SourceError("Expected a JSON object response.")
    return data


def clean_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def short_title(text: str, limit: int = 80) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def parse_timestamp_millis(value: Any) -> str | None:
    if value in (None, ""):
        return None
    try:
        timestamp = float(value) / 1000.0
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def parse_weibo_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    parts = value.split()
    if len(parts) != 6:
        return None
    _, month_label, day, time_part, offset_part, year = parts
    month = WEIBO_MONTHS.get(month_label)
    if month is None:
        return None
    try:
        hour, minute, second = [int(part) for part in time_part.split(":")]
        offset_sign = 1 if offset_part.startswith("+") else -1
        offset_hours = int(offset_part[1:3])
        offset_minutes = int(offset_part[3:5])
        offset = timezone(
            offset_sign * timedelta(hours=offset_hours, minutes=offset_minutes)
        )
        return datetime(
            int(year),
            month,
            int(day),
            hour,
            minute,
            second,
            tzinfo=offset,
        )
    except (TypeError, ValueError):
        return None


def normalize_item(
    *,
    source: str,
    source_type: str,
    title: str,
    url: str | None,
    published_at: str | None = None,
    summary: str | None = None,
    author: str | None = None,
    engagement: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    raw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "source": source,
        "source_type": source_type,
        "title": title,
        "url": url or "",
        "published_at": published_at,
        "summary": summary,
        "author": author,
        "engagement": engagement or {},
        "tags": tags or [],
        "raw": raw or {},
    }


def deduplicate_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in items:
        key = (
            item.get("source", ""),
            item.get("url", ""),
            item.get("title", ""),
            item.get("published_at") or "",
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def build_tavily_query(args: argparse.Namespace) -> str:
    if args.tavily_query:
        return args.tavily_query
    tokens = [part for part in (args.company_name, args.stock_code) if part]
    tokens.extend(["listed company", "announcement", "news", "research"])
    return " ".join(tokens)


def collect_tavily(
    args: argparse.Namespace,
    start_date: str,
    end_date: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query_text = build_tavily_query(args)
    payload: dict[str, Any] = {
        "query": query_text,
        "topic": args.tavily_topic,
        "search_depth": args.tavily_search_depth,
        "max_results": args.max_items,
        "start_date": start_date,
        "end_date": end_date,
        "include_answer": False,
        "include_raw_content": False,
    }
    include_domains = split_csv(args.tavily_include_domains)
    exclude_domains = split_csv(args.tavily_exclude_domains)
    if include_domains:
        payload["include_domains"] = include_domains
    if exclude_domains:
        payload["exclude_domains"] = exclude_domains

    headers = {
        "Authorization": f"Bearer {os.getenv('TAVILY_API_KEY', '').strip()}",
        "User-Agent": args.user_agent,
    }
    diagnostics = {
        "query": query_text,
        "planned_requests": [
            build_plan(
                method="POST",
                url="https://api.tavily.com/search",
                headers=headers,
                json_body=payload,
                auth_env=["TAVILY_API_KEY"],
            )
        ],
    }
    if args.dry_run:
        return [], diagnostics

    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise SourceError("TAVILY_API_KEY is empty.")

    raw_response = http_request(
        method="POST",
        url="https://api.tavily.com/search",
        headers=headers,
        timeout=args.request_timeout,
        json_body=payload,
    )
    data = load_json_payload(raw_response)
    items: list[dict[str, Any]] = []
    for result in data.get("results", [])[: args.max_items]:
        title = clean_text(result.get("title"))
        summary = clean_text(result.get("content"))
        items.append(
            normalize_item(
                source="tavily",
                source_type="web",
                title=title or short_title(summary or query_text),
                url=result.get("url"),
                published_at=result.get("published_date"),
                summary=summary,
                author=result.get("source"),
                engagement={"score": result.get("score")},
                raw=result,
            )
        )
    diagnostics["response_time"] = data.get("response_time")
    diagnostics["result_count"] = len(items)
    return items, diagnostics


def exchange_weibo_code_for_token(
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any] | None]:
    access_token = os.getenv("WEIBO_ACCESS_TOKEN", "").strip()
    if access_token:
        return access_token, None

    if not args.weibo_code:
        raise SourceError(
            "WEIBO_ACCESS_TOKEN is empty. Set the token or pass --weibo-code with app credentials."
        )

    app_key = os.getenv("WEIBO_APP_KEY", "").strip()
    app_secret = os.getenv("WEIBO_APP_SECRET", "").strip()
    redirect_uri = os.getenv("WEIBO_REDIRECT_URI", "").strip()
    if not app_key or not app_secret or not redirect_uri:
        raise SourceError(
            "WEIBO_APP_KEY, WEIBO_APP_SECRET, and WEIBO_REDIRECT_URI are required when using --weibo-code."
        )

    headers = {"User-Agent": args.user_agent}
    form_body = {
        "client_id": app_key,
        "client_secret": app_secret,
        "grant_type": "authorization_code",
        "code": args.weibo_code,
        "redirect_uri": redirect_uri,
    }
    raw_response = http_request(
        method="POST",
        url="https://api.weibo.com/oauth2/access_token",
        headers=headers,
        timeout=args.request_timeout,
        form_body=form_body,
    )
    data = load_json_payload(raw_response)
    token = clean_text(data.get("access_token"))
    if not token:
        raise SourceError("Weibo token exchange succeeded, but no access_token was returned.")
    return token, {
        "uid": data.get("uid"),
        "expires_in": data.get("expires_in"),
    }


def resolve_weibo_uid(
    args: argparse.Namespace,
    access_token: str,
) -> tuple[str, dict[str, Any] | None]:
    if args.weibo_uid:
        return args.weibo_uid, None
    if not args.weibo_screen_name:
        raise SourceError("Provide --weibo-uid or --weibo-screen-name for the Weibo source.")

    headers = {"User-Agent": args.user_agent}
    query = {
        "access_token": access_token,
        "screen_name": args.weibo_screen_name,
    }
    raw_response = http_request(
        method="GET",
        url="https://api.weibo.com/2/users/show.json",
        headers=headers,
        timeout=args.request_timeout,
        query=query,
    )
    data = load_json_payload(raw_response)
    uid = clean_text(data.get("idstr") or data.get("id"))
    if not uid:
        raise SourceError("Weibo user resolution did not return an id.")
    return uid, {
        "screen_name": data.get("screen_name"),
        "verified": data.get("verified"),
    }


def collect_weibo(
    args: argparse.Namespace,
    start_date: str,
    end_date: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    count = min(max(args.max_items * 3, args.max_items), 100)
    diagnostics = {
        "planned_requests": [
            build_plan(
                method="POST",
                url="https://api.weibo.com/oauth2/access_token",
                headers={"User-Agent": args.user_agent},
                form_body={
                    "client_id": os.getenv("WEIBO_APP_KEY", "").strip(),
                    "client_secret": os.getenv("WEIBO_APP_SECRET", "").strip(),
                    "grant_type": "authorization_code",
                    "code": args.weibo_code or "",
                    "redirect_uri": os.getenv("WEIBO_REDIRECT_URI", "").strip(),
                },
                auth_env=["WEIBO_APP_KEY", "WEIBO_APP_SECRET", "WEIBO_REDIRECT_URI"],
            ),
            build_plan(
                method="GET",
                url="https://api.weibo.com/2/users/show.json",
                headers={"User-Agent": args.user_agent},
                query={
                    "access_token": os.getenv("WEIBO_ACCESS_TOKEN", "").strip(),
                    "screen_name": args.weibo_screen_name or "",
                },
                auth_env=["WEIBO_ACCESS_TOKEN"],
            ),
            build_plan(
                method="GET",
                url="https://api.weibo.com/2/statuses/user_timeline.json",
                headers={"User-Agent": args.user_agent},
                query={
                    "access_token": os.getenv("WEIBO_ACCESS_TOKEN", "").strip(),
                    "uid": args.weibo_uid or "",
                    "count": count,
                    "trim_user": 0,
                },
                auth_env=["WEIBO_ACCESS_TOKEN"],
            ),
        ]
    }
    if args.dry_run:
        return [], diagnostics

    access_token, token_info = exchange_weibo_code_for_token(args)
    resolved_uid, user_info = resolve_weibo_uid(args, access_token)
    query = {
        "access_token": access_token,
        "uid": resolved_uid,
        "count": count,
        "trim_user": 0,
    }
    raw_response = http_request(
        method="GET",
        url="https://api.weibo.com/2/statuses/user_timeline.json",
        headers={"User-Agent": args.user_agent},
        timeout=args.request_timeout,
        query=query,
    )
    data = load_json_payload(raw_response)

    start_boundary = datetime.fromisoformat(f"{start_date}T00:00:00+00:00").date()
    end_boundary = datetime.fromisoformat(f"{end_date}T23:59:59+00:00").date()
    items: list[dict[str, Any]] = []
    for status in data.get("statuses", []):
        published_dt = parse_weibo_datetime(status.get("created_at"))
        if published_dt is not None:
            published_date = published_dt.astimezone(timezone.utc).date()
            if published_date < start_boundary or published_date > end_boundary:
                continue
            published_at = published_dt.astimezone(timezone.utc).isoformat()
        else:
            published_at = None

        text = clean_text(status.get("text"))
        user = status.get("user") or {}
        uid = clean_text(user.get("idstr") or user.get("id") or resolved_uid)
        bid = clean_text(status.get("bid"))
        item_url = f"https://weibo.com/{uid}/{bid}" if uid and bid else ""
        items.append(
            normalize_item(
                source="weibo",
                source_type="social",
                title=short_title(text or "Weibo status"),
                url=item_url,
                published_at=published_at,
                summary=text,
                author=clean_text(user.get("screen_name")),
                engagement={
                    "reposts_count": status.get("reposts_count"),
                    "comments_count": status.get("comments_count"),
                    "attitudes_count": status.get("attitudes_count"),
                },
                raw=status,
            )
        )
        if len(items) >= args.max_items:
            break

    diagnostics["token_info"] = token_info or {}
    diagnostics["user_info"] = user_info or {}
    diagnostics["resolved_uid"] = resolved_uid
    diagnostics["result_count"] = len(items)
    return items, diagnostics


def build_cninfo_payload(
    args: argparse.Namespace,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    return {
        "pageNum": 1,
        "pageSize": args.max_items,
        "column": args.cninfo_column,
        "tabName": "fulltext",
        "plate": "",
        "stock": args.cninfo_stock or args.stock_code or "",
        "searchkey": args.company_name or args.stock_code or "",
        "secid": "",
        "category": args.cninfo_category,
        "trade": "",
        "seDate": f"{start_date}~{end_date}",
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }


def collect_cninfo(
    args: argparse.Namespace,
    start_date: str,
    end_date: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    headers = {
        "User-Agent": args.user_agent,
        "Referer": "https://www.cninfo.com.cn/",
        "X-Requested-With": "XMLHttpRequest",
    }
    form_body = build_cninfo_payload(args, start_date, end_date)
    diagnostics = {
        "planned_requests": [
            build_plan(
                method="POST",
                url="https://www.cninfo.com.cn/new/hisAnnouncement/query",
                headers=headers,
                form_body=form_body,
            )
        ]
    }
    if args.dry_run:
        return [], diagnostics

    raw_response = http_request(
        method="POST",
        url="https://www.cninfo.com.cn/new/hisAnnouncement/query",
        headers=headers,
        timeout=args.request_timeout,
        form_body=form_body,
    )
    data = load_json_payload(raw_response)
    items: list[dict[str, Any]] = []
    for announcement in data.get("announcements", [])[: args.max_items]:
        relative_url = announcement.get("adjunctUrl") or ""
        item_url = relative_url
        if relative_url and not relative_url.startswith("http"):
            item_url = parse.urljoin("https://static.cninfo.com.cn/", relative_url)
        title = clean_text(announcement.get("announcementTitle"))
        tags = [
            clean_text(announcement.get("announcementTypeName")),
            clean_text(args.cninfo_column),
        ]
        items.append(
            normalize_item(
                source="cninfo",
                source_type="official_announcement",
                title=title or "CNINFO announcement",
                url=item_url,
                published_at=parse_timestamp_millis(announcement.get("announcementTime")),
                summary=clean_text(announcement.get("announcementTypeName")),
                author=clean_text(announcement.get("secName")),
                tags=[tag for tag in tags if tag],
                raw=announcement,
            )
        )
    diagnostics["total_records"] = data.get("totalRecordNum")
    diagnostics["result_count"] = len(items)
    return items, diagnostics


def build_sse_query(
    args: argparse.Namespace,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    callback = f"jsonpCallback{int(time.time() * 1000)}"
    query = {
        "jsonCallBack": callback,
        "productId": args.stock_code or "",
        "reportType": args.sse_report_type,
        "beginDate": start_date,
        "endDate": end_date,
        "pageHelp.pageSize": args.max_items,
        "pageHelp.pageNo": 1,
        "pageHelp.beginPage": 1,
        "pageHelp.endPage": 1,
        "_": int(time.time() * 1000),
    }
    if args.sse_report_type2:
        query["reportType2"] = args.sse_report_type2
    return query


def collect_sse(
    args: argparse.Namespace,
    start_date: str,
    end_date: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not args.stock_code:
        raise SourceError("--stock-code is required for the SSE source.")

    headers = {
        "User-Agent": args.user_agent,
        "Referer": "https://www.sse.com.cn/",
    }
    query = build_sse_query(args, start_date, end_date)
    diagnostics = {
        "planned_requests": [
            build_plan(
                method="GET",
                url="https://query.sse.com.cn/security/stock/queryCompanyBulletin.do",
                headers=headers,
                query=query,
            )
        ]
    }
    if args.dry_run:
        return [], diagnostics

    raw_response = http_request(
        method="GET",
        url="https://query.sse.com.cn/security/stock/queryCompanyBulletin.do",
        headers=headers,
        timeout=args.request_timeout,
        query=query,
    )
    data = load_json_payload(raw_response)
    raw_items = data.get("result") or data.get("pageHelp", {}).get("data") or []
    items: list[dict[str, Any]] = []
    for announcement in raw_items[: args.max_items]:
        relative_url = announcement.get("URL") or announcement.get("url") or ""
        item_url = relative_url
        if relative_url and not relative_url.startswith("http"):
            item_url = parse.urljoin("https://www.sse.com.cn", relative_url)
        title = clean_text(announcement.get("TITLE") or announcement.get("title"))
        bulletin_type = clean_text(
            announcement.get("BULLETIN_TYPE") or announcement.get("bulletinType")
        )
        items.append(
            normalize_item(
                source="sse",
                source_type="official_announcement",
                title=title or "SSE announcement",
                url=item_url,
                published_at=clean_text(
                    announcement.get("SSEDATE") or announcement.get("publishDate")
                )
                or None,
                summary=bulletin_type or None,
                author=clean_text(
                    announcement.get("SECURITY_ABBR") or announcement.get("companyCode")
                ),
                tags=[
                    value
                    for value in (
                        bulletin_type,
                        clean_text(args.sse_report_type),
                        clean_text(args.sse_report_type2),
                    )
                    if value
                ],
                raw=announcement,
            )
        )
    diagnostics["result_count"] = len(items)
    return items, diagnostics


COLLECTORS = {
    "tavily": collect_tavily,
    "weibo": collect_weibo,
    "cninfo": collect_cninfo,
    "sse": collect_sse,
}


def run() -> int:
    args = parse_args()
    sources = parse_sources(args.sources)
    start_date, end_date = compute_window(args.days)

    output: dict[str, Any] = {
        "company": {
            "company_name": args.company_name,
            "stock_code": args.stock_code,
            "weibo_uid": args.weibo_uid,
            "weibo_screen_name": args.weibo_screen_name,
            "cninfo_stock": args.cninfo_stock,
        },
        "window": {
            "start_date": start_date,
            "end_date": end_date,
            "days": args.days,
        },
        "requested_sources": sources,
        "collected_at": utc_now(),
        "items": [],
        "errors": [],
        "diagnostics": {},
    }

    collected: list[dict[str, Any]] = []
    for source in sources:
        collector = COLLECTORS[source]
        try:
            items, diagnostics = collector(args, start_date, end_date)
            output["diagnostics"][source] = diagnostics
            collected.extend(items)
        except SourceError as exc:
            output["errors"].append({"source": source, "message": str(exc)})

    output["items"] = deduplicate_items(collected)

    serialized = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        target = Path(args.output)
        ensure_output_dir(target)
        target.write_text(serialized, encoding="utf-8")
    else:
        print(serialized)

    if args.fail_on_source_error and output["errors"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run())
