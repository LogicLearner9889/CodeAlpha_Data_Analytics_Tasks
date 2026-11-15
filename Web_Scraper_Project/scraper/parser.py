import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .exceptions import ParseError, StructureChangedError


def _soup(html: str) -> BeautifulSoup:
    if not html or not html.strip():
        raise ParseError("Empty HTML")
    try:
        return BeautifulSoup(html, "html.parser")
    except Exception as exc:  # bs4 rarely raises, but never trust input
        raise ParseError(f"HTML could not be parsed: {exc}") from exc


def _text(node):
    return node.get_text(strip=True) if node else None


def parse_listing(html: str, page_url: str):
    """Return (items, next_page_url). Raises StructureChangedError if layout is unrecognised."""
    soup = _soup(html)
    cards = soup.select("article.product_pod")
    if not cards:
        # Distinguish "empty page" from "layout changed"
        if soup.select_one("ul.pager, section, div.page_inner") is None:
            raise StructureChangedError(f"No product container or page markup found at {page_url}")
        return [], None

    items, skipped = [], 0
    for card in cards:
        try:
            link = card.select_one("h3 a")
            if link is None or not link.get("href"):
                raise ValueError("missing title link")
            rating_node = card.select_one("p.star-rating")
            classes = (rating_node.get("class") or []) if rating_node else []
            rating_class = [c for c in classes if c != "star-rating"]
            items.append({
                "title": link.get("title") or _text(link),
                "url": urljoin(page_url, link["href"]),
                "price_raw": _text(card.select_one("p.price_color")),
                "rating_raw": rating_class[0] if rating_class else None,
                "availability_raw": _text(card.select_one("p.availability")),
            })
        except (ValueError, KeyError, AttributeError):
            skipped += 1  # one malformed card must not kill the page
    if skipped and not items:
        raise StructureChangedError(f"All {skipped} product cards were malformed at {page_url}")

    nxt = soup.select_one("li.next a")
    next_url = urljoin(page_url, nxt["href"]) if nxt and nxt.get("href") else None
    return items, next_url


def parse_detail(html: str) -> dict:
    """Extract extra fields from a product page. Missing optional fields become None."""
    soup = _soup(html)
    if soup.select_one("div.product_main") is None:
        raise StructureChangedError("Product page layout not recognised")

    table = {}
    for row in soup.select("table.table-striped tr"):
        th, td = row.find("th"), row.find("td")
        if th and td:
            table[_text(th)] = _text(td)

    crumbs = soup.select("ul.breadcrumb li a")
    category = _text(crumbs[2]) if len(crumbs) >= 3 else None

    desc_header = soup.select_one("#product_description")
    desc = _text(desc_header.find_next_sibling("p")) if desc_header else None

    avail = table.get("Availability") or _text(soup.select_one("p.availability")) or ""
    stock_match = re.search(r"(\d+)\s+available", avail)
    return {
        "upc": table.get("UPC"),
        "category": category,
        "description": desc,
        "stock_count": int(stock_match.group(1)) if stock_match else None,
        "num_reviews": table.get("Number of reviews"),
    }
