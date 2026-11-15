# Web Scraping Project with Full Error Handling

Scrapes the book catalogue at <https://books.toscrape.com> (a sandbox site built for scraping practice)
into a clean, typed dataset: title, price, rating, stock, category, UPC, description.

## Setup
```bash
pip install -r requirements.txt
python main.py --max-pages 3          # polite default: 1 request/second
python main.py --max-pages 50 --delay 1.5 -v
python main.py --resume               # continue after a crash / Ctrl+C
python main.py --no-details           # fast mode: listing pages only
python -m unittest discover -s . -p "test_all.py" -v
```
Edit `scraper/config.py` to set your own User-Agent contact email, timeouts and retry counts.

## Output (`output/`)
| File | Purpose |
|---|---|
| `books.csv` / `books.json` / `books.db` | Final datasets (CSV opens in Excel; SQLite for SQL analysis) |
| `books.jsonl` | Append-only crawl log, fsynced per record (crash-safe) |
| `failures.csv` | Every URL that failed, at which stage, and why |
| `checkpoint.json` | Resume pointer |
| `scraper.log` | Full rotating debug log |

## Error-handling map
| Problem | Handling |
|---|---|
| Timeouts, connection reset/refused, chunked-encoding errors | Retry with exponential backoff (separate connect/read timeouts) |
| HTTP 429 / 5xx / 408 | Retry, honouring `Retry-After` |
| HTTP 404 / 410 | No retry; on a listing page it means end of pagination |
| HTTP 401 / 403 / 400 | Fail fast with `HTTPStatusError` |
| Redirect loops, malformed URLs | `FetchError`, no retry |
| Non-HTML, empty, or oversized (>5 MB) responses | `InvalidContentError` |
| Wrong character encoding (`Â£`) | UTF-8 first, fall back to declared / latin-1 |
| robots.txt disallow (or 403 on robots.txt) | `RobotsDisallowedError`; robots fetched once per host |
| Site layout changed / captcha page | `StructureChangedError` instead of silent empty data |
| One malformed product card | Skipped; rest of page still parsed |
| Missing optional fields | `None`, never a crash |
| Bad price/title/URL | `ValidationError`, logged to `failures.csv` |
| Detail page fails | Listing-level data is **saved as a partial record** (`detail_ok=False`) |
| Duplicates across pages | De-duplicated by URL |
| Many consecutive failures (likely blocked) | Circuit breaker stops the crawl, still exports data |
| Unexpected bug on one item | Caught, logged with traceback, crawl continues |
| Ctrl+C / crash | Data already on disk; exports run in `finally`; `--resume` continues |
| Disk write failure during export | Atomic temp-file writes; JSONL remains intact |

Exit codes: `0` clean, `1` nothing saved, `2` cannot create output dir, `3` finished with some failures.

## Adapting to another site
1. Change `base_url`/`start_path` in `config.py`.
2. Rewrite the CSS selectors in `scraper/parser.py` (inspect the page with browser DevTools -> F12).
3. Adjust fields in `validator.py` and `storage.py`.
The HTTP, retry, resume, and storage layers need no changes.

## Responsible scraping
Check the site's Terms of Service and `robots.txt`, keep the delay at 1s or more, identify yourself in the
User-Agent, and never scrape personal data or content behind a login without permission.
No-code alternatives: Octoparse and ParseHub can export to the same CSV format.
