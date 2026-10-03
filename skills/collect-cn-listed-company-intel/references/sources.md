# Source Notes

## Contents

1. Tavily
2. Weibo Open API
3. CNINFO
4. SSE
5. Extension Notes

## Tavily

- Endpoint used by the script: `POST https://api.tavily.com/search`
- Auth: `TAVILY_API_KEY`
- Best use: broad company coverage, finance/news discovery, and link expansion beyond exchange sites
- Default script choices:
  - `topic=finance`
  - `search_depth=advanced`
  - `start_date` and `end_date` follow `--days`
- Prefer `--tavily-query` when the company name is ambiguous or when you want product-line filters.

## Weibo Open API

- Endpoints used by the script:
  - `POST https://api.weibo.com/oauth2/access_token`
  - `GET https://api.weibo.com/2/users/show.json`
  - `GET https://api.weibo.com/2/statuses/user_timeline.json`
- Auth:
  - Preferred: `WEIBO_ACCESS_TOKEN`
  - Optional token exchange path: `WEIBO_APP_KEY`, `WEIBO_APP_SECRET`, `WEIBO_REDIRECT_URI`, plus `--weibo-code`
- Best use: company-account posts, management commentary, campaign messaging, and fast-moving social signal checks
- Constraints:
  - The script does not open the interactive browser login flow.
  - Account scopes and app approval can limit endpoint access.
  - For precision, pass `--weibo-uid` or `--weibo-screen-name`; do not rely on fuzzy company-name matching.

## CNINFO

- Endpoint used by the script: `POST https://www.cninfo.com.cn/new/hisAnnouncement/query`
- Auth: public endpoint, no API key in this skill
- Best use: official announcement retrieval for CNINFO-backed disclosure workflows
- Important parameters:
  - `stock`: best when it is the exact CNINFO token used by the site
  - `searchkey`: fallback company-name or stock-code search term
  - `column`: script default is `szse`
  - `seDate`: rolling window in `YYYY-MM-DD~YYYY-MM-DD`
- Practical note:
  - The site often works more reliably when `stock` is the exact token rather than a bare stock code.
  - If recall is poor, inspect the current site request shape and pass `--cninfo-stock`.

## SSE

- Endpoint used by the script: `GET https://query.sse.com.cn/security/stock/queryCompanyBulletin.do`
- Auth: public endpoint, no API key in this skill
- Best use: official announcements for Shanghai-listed companies and STAR Market issuers
- Important parameters:
  - `productId`: stock code
  - `reportType`: primary bulletin filter
  - `reportType2`: secondary bulletin filter
  - `beginDate`, `endDate`: rolling window
- Practical note:
  - The endpoint is the page-backed query interface used by the official SSE site and may return JSONP.
  - Keep the `Referer` header; some public page-backed endpoints are stricter without it.

## Extension Notes

- The collector normalizes all sources into one JSON shape so downstream sentiment labeling can be
  source-agnostic.
- Add new official sources by following the same pattern:
  1. keep raw request parameters visible in `planned_requests`
  2. preserve source URLs and timestamps
  3. keep the source-specific raw payload in `raw`
