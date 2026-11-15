import json
import logging
import shutil
import tempfile
import unittest
from pathlib import Path
from scraper.config import Config
from scraper.crawler import Crawler
from scraper.exceptions import (
    HTTPStatusError, InvalidContentError, PageNotFoundError, ParseError,
    RetryExhaustedError, RobotsDisallowedError, StructureChangedError, ValidationError,
)
from scraper.http_client import HttpClient
from scraper.parser import parse_detail, parse_listing
from scraper.storage import Storage
from scraper.validator import clean_record
from tests import fake_server

LOG = logging.getLogger("test")
LOG.addHandler(logging.NullHandler())


def make_cfg(base, tmp, **kw):
    defaults = dict(base_url=base + "/", delay=0, backoff_factor=0.01, max_retries=3,
                    connect_timeout=2, read_timeout=0.5, output_dir=Path(tmp))
    defaults.update(kw)
    return Config(**defaults)


class ParserTests(unittest.TestCase):
    def test_listing_ok(self):
        html = fake_server.listing(1, [("a", "A", "1.00")], last=False)
        items, nxt = parse_listing(html, "http://x/catalogue/page-1.html")
        self.assertEqual(items[0]["url"], "http://x/catalogue/a/index.html")
        self.assertEqual(nxt, "http://x/catalogue/page-2.html")

    def test_malformed_card_skipped(self):
        html = "<section>" + fake_server.CARD.format(slug="a", title="A", price="1") + \
               '<article class="product_pod"><h3>no link</h3></article></section>'
        items, _ = parse_listing(html, "http://x/")
        self.assertEqual(len(items), 1)

    def test_layout_change_detected(self):
        with self.assertRaises(StructureChangedError):
            parse_listing("<html><body><p>Access denied</p></body></html>", "http://x/")

    def test_empty_html(self):
        with self.assertRaises(ParseError):
            parse_listing("", "http://x/")

    def test_detail_ok_and_missing_optional(self):
        d = parse_detail(fake_server.DETAIL.format(title="T", slug="s"))
        self.assertEqual((d["category"], d["stock_count"], d["upc"]), ("Poetry", 7, "upc-s"))
        d = parse_detail('<div class="product_main"></div>')
        self.assertIsNone(d["upc"])

    def test_detail_wrong_page(self):
        with self.assertRaises(StructureChangedError):
            parse_detail("<html><body>captcha</body></html>")


class ValidatorTests(unittest.TestCase):
    base = {"title": "T", "url": "http://x/t", "price_raw": "£51.77", "rating_raw": "Three",
            "availability_raw": "In stock"}

    def test_clean(self):
        r = clean_record(dict(self.base))
        self.assertEqual((r["price_gbp"], r["rating"], r["in_stock"]), (51.77, 3, True))

    def test_bad_inputs(self):
        for patch in ({"title": ""}, {"url": "ftp://x"}, {"price_raw": "free"}, {"price_raw": "£9999999"}):
            with self.assertRaises(ValidationError):
                clean_record({**self.base, **patch})


class HttpClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server, cls.base = fake_server.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        fake_server.State.hits.clear()
        self.tmp = tempfile.mkdtemp()
        self.client = HttpClient(make_cfg(self.base, self.tmp), LOG)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_success_and_utf8(self):
        self.assertIn("£", self.client.get(self.base + "/ok"))

    def test_retry_on_503_then_success(self):
        self.assertIn("ok", self.client.get(self.base + "/flaky"))
        self.assertEqual(fake_server.State.hits["/flaky"], 3)

    def test_429_retry_after(self):
        self.assertIn("ok", self.client.get(self.base + "/ratelimited"))

    def test_retries_exhausted(self):
        with self.assertRaises(RetryExhaustedError):
            self.client.get(self.base + "/always503")
        self.assertEqual(fake_server.State.hits["/always503"], 3)

    def test_404_not_retried(self):
        with self.assertRaises(PageNotFoundError):
            self.client.get(self.base + "/nope")
        self.assertEqual(fake_server.State.hits["/nope"], 1)

    def test_403_not_retried(self):
        with self.assertRaises(HTTPStatusError):
            self.client.get(self.base + "/forbidden")
        self.assertEqual(fake_server.State.hits["/forbidden"], 1)

    def test_non_html_and_empty(self):
        with self.assertRaises(InvalidContentError):
            self.client.get(self.base + "/image")
        with self.assertRaises(InvalidContentError):
            self.client.get(self.base + "/empty")

    def test_timeout_retries_then_fails(self):
        with self.assertRaises(RetryExhaustedError):
            self.client.get(self.base + "/slow")

    def test_robots_blocks(self):
        with self.assertRaises(RobotsDisallowedError):
            self.client.get(self.base + "/private")

    def test_connection_refused(self):
        c = HttpClient(make_cfg("http://127.0.0.1:1", self.tmp, respect_robots=False), LOG)
        with self.assertRaises(RetryExhaustedError):
            c.get("http://127.0.0.1:1/x")

    def test_invalid_url(self):
        from scraper.exceptions import FetchError
        c = HttpClient(make_cfg(self.base, self.tmp, respect_robots=False), LOG)
        with self.assertRaises(FetchError):
            c.get("not-a-url")


class CrawlerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server, cls.base = fake_server.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        fake_server.State.hits.clear()
        fake_server.State.broken_details.clear()
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_crawl(self, **kw):
        cfg = make_cfg(self.base, self.tmp, read_timeout=2, **kw)
        crawler = Crawler(cfg, HttpClient(cfg, LOG), Storage(cfg.output_dir), LOG)
        return crawler, crawler.run()

    def test_full_run_dedupes_and_stops_on_404(self):
        _, stats = self.run_crawl(max_pages=10)
        self.assertEqual((stats["saved"], stats["duplicates"], stats["pages"]), (3, 1, 2))
        for name in ("books.csv", "books.json", "books.db"):
            self.assertTrue((Path(self.tmp) / name).exists(), name)
        self.assertEqual(len(json.loads((Path(self.tmp) / "books.json").read_text())), 3)

    def test_partial_record_when_detail_breaks(self):
        fake_server.State.broken_details.add("book-b")
        c, stats = self.run_crawl(max_pages=1, max_retries=2)
        self.assertEqual((stats["saved"], stats["partial"]), (1, 1))
        bad = [r for r in c.records if not r["detail_ok"]]
        self.assertEqual(bad[0]["title"], "Book B")
        self.assertTrue((Path(self.tmp) / "failures.csv").exists())

    def test_circuit_breaker(self):
        fake_server.State.broken_details.update({"book-a", "book-b", "book-c"})
        _, stats = self.run_crawl(max_pages=5, max_retries=1, max_consecutive_failures=2)
        self.assertEqual(stats["partial"], 2)           # stopped after 2 consecutive failures
        self.assertTrue((Path(self.tmp) / "books.csv").exists())  # still exported

    def test_resume(self):
        self.run_crawl(max_pages=1)
        _, stats = self.run_crawl(max_pages=5, resume=True)
        self.assertEqual(stats["pages"], 1)             # only page 2 processed (then 404)
        self.assertEqual(len(json.loads((Path(self.tmp) / "books.json").read_text())), 3)

    def test_corrupt_jsonl_tail_ignored(self):
        self.run_crawl(max_pages=1)
        with open(Path(self.tmp) / "books.jsonl", "a") as fh:
            fh.write('{"title": "half-writ')
        self.assertEqual(len(Storage(Path(self.tmp)).load_records()), 2)

    def test_dead_server_exits_gracefully(self):
        cfg = make_cfg("http://127.0.0.1:1", self.tmp, respect_robots=False, max_retries=2)
        stats = Crawler(cfg, HttpClient(cfg, LOG), Storage(cfg.output_dir), LOG).run()
        self.assertEqual((stats["saved"], stats["failed"]), (0, 1))


if __name__ == "__main__":
    unittest.main()
