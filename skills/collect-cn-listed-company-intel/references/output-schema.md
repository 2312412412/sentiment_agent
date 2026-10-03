# Output Schema

## Top-Level Keys

- `company`: input identifiers used for collection
- `window`: requested date window
- `requested_sources`: source list actually attempted
- `collected_at`: UTC timestamp for the run
- `items`: normalized evidence items
- `errors`: source-level failures that did not crash the whole run
- `diagnostics`: request plans and source-specific metadata

## Normalized Item Shape

Each element in `items[]` uses this structure:

```json
{
  "source": "tavily",
  "source_type": "web",
  "title": "Example title",
  "url": "https://example.com/article",
  "published_at": "2026-03-20T08:30:00+00:00",
  "summary": "Short normalized snippet or announcement type.",
  "author": "Publisher or account name",
  "engagement": {
    "score": 0.81
  },
  "tags": ["finance"],
  "raw": {}
}
```

## Source-Type Conventions

- `web`: broad web or news discovery results
- `social`: social-account posts and engagement fields
- `official_announcement`: exchange or disclosure-platform announcements

## Downstream Usage

- Use `items[]` as the evidence table for sentiment, event extraction, and timeline alignment.
- Use `errors[]` to decide whether a collection run should be retried.
- Use `diagnostics.<source>.planned_requests` when a source starts failing after a site or API
  change.
- Do not drop `raw` during the first normalization stage; it is the cheapest recovery path when a
  downstream parser needs a source-specific field later.
