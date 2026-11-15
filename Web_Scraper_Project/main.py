import argparse
import sys
from pathlib import Path

from scraper.config import Config
from scraper.crawler import Crawler
from scraper.http_client import HttpClient
from scraper.logger import setup_logger
from scraper.storage import Storage


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Book catalogue scraper (books.toscrape.com)")
    p.add_argument("--max-pages", type=int, default=3)
    p.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    p.add_argument("--no-details", action="store_true", help="skip per-book detail pages")
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--resume", action="store_true", help="continue from last checkpoint")
    p.add_argument("--ignore-robots", action="store_true", help="NOT recommended")
    p.add_argument("--user-agent", default=None)
    p.add_argument("--base-url", default=None, help="override target site (e.g. a local mirror)")
    p.add_argument("--start-path", default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    if args.max_pages < 1:
        p.error("--max-pages must be >= 1")
    if args.delay < 0:
        p.error("--delay must be >= 0")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = Config(max_pages=args.max_pages, delay=args.delay, fetch_details=not args.no_details,
                 output_dir=args.output_dir, resume=args.resume,
                 respect_robots=not args.ignore_robots)
    if args.user_agent:
        cfg.user_agent = args.user_agent
    if args.base_url:
        cfg.base_url = args.base_url.rstrip("/") + "/"
    if args.start_path:
        cfg.start_path = args.start_path

    try:
        log = setup_logger(cfg.output_dir, args.verbose)
        storage = Storage(cfg.output_dir)
    except OSError as exc:
        print(f"Cannot create output directory: {exc}", file=sys.stderr)
        return 2

    stats = Crawler(cfg, HttpClient(cfg, log), storage, log).run()
    log.info("Done: %s", stats)
    # Exit codes: 0 ok, 1 nothing saved, 3 completed with some failures
    if stats["saved"] + stats["partial"] == 0:
        return 1
    return 3 if (stats["failed"] or stats["partial"]) else 0


if __name__ == "__main__":
    sys.exit(main())
