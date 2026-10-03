---
name: collect-cn-listed-company-intel
description: Collect raw intelligence for Chinese listed companies from Tavily, Weibo Open API, and official disclosure portals such as CNINFO and SSE. Use when Codex needs recent company announcements, social posts, or source-linked evidence for A-share research, monitoring, event tracking, or sentiment analysis.
---

# Collect CN Listed Company Intel

## Overview

Use this skill to gather raw, source-linked evidence before any summarization or sentiment work.
Prefer collecting timestamps, URLs, issuer metadata, and original snippets first, then score or
label the material in a later step.

## Quick Start

1. Choose identifiers for the company:
   - `--company-name` for Tavily and human-readable output.
   - `--stock-code` for SSE and most official-announcement workflows.
   - `--weibo-uid` or `--weibo-screen-name` for Weibo.
   - `--cninfo-stock` when you already know the exact CNINFO `stock` token.
2. Export credentials only for the sources you need. Leave unused keys empty.
   - `TAVILY_API_KEY=""`
   - `WEIBO_ACCESS_TOKEN=""`
   - `WEIBO_APP_KEY=""`
   - `WEIBO_APP_SECRET=""`
   - `WEIBO_REDIRECT_URI=""`
3. Run the collector script:

```bash
python scripts/fetch_company_intel.py \
  --company-name "Kweichow Moutai" \
  --stock-code 600519 \
  --sources tavily,sse \
  --days 30
```

## Source Selection

- Use `tavily` for broad web coverage, finance/news discovery, and links outside exchange portals.
- Use `weibo` for company-account or management-account social posts.
- Use `sse` for Shanghai Stock Exchange or STAR Market announcements.
- Use `cninfo` for CNINFO-backed disclosure searches, especially SZSE/ChiNext flows.
- Mix at least one web source and one official disclosure source when building sentiment or event
  datasets.

## Workflow

1. Collect raw items with `scripts/fetch_company_intel.py`.
2. Inspect `errors[]` in the JSON result and rerun missing sources with corrected auth or
   identifiers.
3. Keep the raw JSON result alongside any derived sentiment labels or summaries.
4. Only after retrieval, run event extraction, sentiment scoring, or portfolio logic.

## Script Usage

Common patterns:

```bash
python scripts/fetch_company_intel.py \
  --company-name "Contemporary Amperex Technology" \
  --stock-code 300750 \
  --sources tavily,cninfo \
  --days 14 \
  --output outputs/catl.json
```

```bash
python scripts/fetch_company_intel.py \
  --company-name "BYD" \
  --stock-code 002594 \
  --weibo-screen-name "BYD Auto" \
  --sources tavily,weibo,cninfo \
  --days 7
```

```bash
python scripts/fetch_company_intel.py \
  --company-name "Kweichow Moutai" \
  --stock-code 600519 \
  --sources sse \
  --dry-run
```

Key flags:

- `--sources`: comma-separated subset of `tavily,weibo,cninfo,sse`
- `--days`: rolling lookback window
- `--max-items`: per-source item limit
- `--output`: write normalized JSON to disk instead of stdout
- `--dry-run`: show planned requests without making network calls
- `--fail-on-source-error`: return a non-zero exit code if any requested source fails

## Weibo Authentication

Prefer passing a ready `WEIBO_ACCESS_TOKEN`.
If an authorization code is already available, pass `--weibo-code` and export:

- `WEIBO_APP_KEY=""`
- `WEIBO_APP_SECRET=""`
- `WEIBO_REDIRECT_URI=""`

The script can exchange the code for an access token, but it does not drive the browser-based login
flow itself.

## Official Announcement Notes

- `sse` uses the public announcement query endpoint behind the official SSE site.
- `cninfo` uses the public CNINFO announcement search endpoint. Precision is best when
  `--cninfo-stock` contains the exact CNINFO token used by the site; otherwise the script falls back
  to `--stock-code` and `--company-name`.
- These web-backed endpoints can change when the exchange site changes. When a source breaks, first
  inspect the site request shape before changing downstream logic.

Read `references/sources.md` for source-specific auth, endpoint, and field notes.
Read `references/output-schema.md` before wiring the collector into downstream sentiment pipelines.
