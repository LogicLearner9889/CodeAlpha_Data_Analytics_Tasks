import re
from datetime import datetime, timezone

from .exceptions import ValidationError

RATINGS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def clean_record(raw: dict) -> dict:
    title = (raw.get("title") or "").strip()
    url = (raw.get("url") or "").strip()
    if not title:
        raise ValidationError(f"Missing title for {url or 'unknown url'}")
    if not url.startswith(("http://", "https://")):
        raise ValidationError(f"Invalid URL: {url!r}")

    price_match = re.search(r"\d+(?:[.,]\d+)?", raw.get("price_raw") or "")
    if not price_match:
        raise ValidationError(f"Unparseable price {raw.get('price_raw')!r} for '{title}'")
    price = float(price_match.group().replace(",", "."))
    if price < 0 or price > 100_000:
        raise ValidationError(f"Price out of range ({price}) for '{title}'")

    rating = RATINGS.get(raw.get("rating_raw"))
    in_stock = "in stock" in (raw.get("availability_raw") or "").lower()

    num_reviews = raw.get("num_reviews")
    try:
        num_reviews = int(num_reviews) if num_reviews not in (None, "") else None
    except ValueError:
        num_reviews = None

    return {
        "title": title,
        "url": url,
        "upc": raw.get("upc"),
        "category": raw.get("category"),
        "price_gbp": price,
        "rating": rating,
        "in_stock": in_stock,
        "stock_count": raw.get("stock_count"),
        "num_reviews": num_reviews,
        "description": (raw.get("description") or "").strip() or None,
        "detail_ok": raw.get("detail_ok", False),
        "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
