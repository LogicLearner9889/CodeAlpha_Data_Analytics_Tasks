import logging
from .config import Config
from .exceptions import (
    CircuitBreakerOpen, FetchError, PageNotFoundError, ParseError, ValidationError,
)
from .http_client import HttpClient
from .parser import parse_detail, parse_listing
from .storage import Storage
from .validator import clean_record


class Crawler:
    def __init__(self, cfg: Config, client: HttpClient, storage: Storage, logger: logging.Logger):
        self.cfg, self.client, self.storage, self.log = cfg, client, storage, logger
        self.records = []
        self.seen = set()
        self.stats = {"pages": 0, "saved": 0, "partial": 0, "failed": 0, "duplicates": 0}
        self._consecutive_failures = 0

    #run
    def run(self) -> dict:
        url, pages_done = self._start_state()
        try:
            while url and self.stats["pages"] < self.cfg.max_pages:
                self.log.info("Page %d: %s", pages_done + 1, url)
                items, next_url = self._fetch_listing(url)
                if items is None:        # listing failed; checkpoint still points here
                    break
                for item in items:
                    self._process_item(item)
                pages_done += 1
                self.stats["pages"] += 1
                self.storage.save_checkpoint(next_url, pages_done)
                url = next_url
        except CircuitBreakerOpen as exc:
            self.log.error("Circuit breaker tripped: %s", exc)
        except KeyboardInterrupt:
            self.log.warning("Interrupted - saving what we have")
        finally:
            self._finalize()
        return self.stats

    #helpers
    def _start_state(self):
        if self.cfg.resume:
            cp = self.storage.load_checkpoint()
            self.records = self.storage.load_records()
            self.seen = {r["url"] for r in self.records}
            if cp and cp.get("next_url"):
                self.log.info("Resuming from %s with %d existing records",
                              cp["next_url"], len(self.records))
                return cp["next_url"], cp.get("pages_done", 0)
            self.log.info("No usable checkpoint - starting fresh (keeping %d records)",
                          len(self.records))
        else:
            self.storage.reset()
        return self.cfg.start_url, 0

    def _fetch_listing(self, url):
        try:
            html = self.client.get(url)
            return parse_listing(html, url)
        except PageNotFoundError:
            self.log.info("Reached end of pagination (404): %s", url)
            self.storage.save_checkpoint(None, self.stats["pages"])
            return None, None
        except (FetchError, ParseError) as exc:
            self.log.error("Listing page failed, stopping pagination: %s", exc)
            self.storage.log_failure(url, "listing", str(exc))
            self.stats["failed"] += 1
            return None, None

    def _process_item(self, item: dict):
        if item["url"] in self.seen:
            self.stats["duplicates"] += 1
            return
        try:
            partial = False
            if self.cfg.fetch_details:
                try:
                    item.update(parse_detail(self.client.get(item["url"])))
                    item["detail_ok"] = True
                    self._consecutive_failures = 0
                except (FetchError, ParseError) as exc:
                    # Keep the listing-level data instead of losing the whole record
                    partial = True
                    self.log.warning("Detail failed (saving partial): %s", exc)
                    self.storage.log_failure(item["url"], "detail", str(exc))
                    self._consecutive_failures += 1
            record = clean_record(item)
            self.storage.append(record)
            self.records.append(record)
            self.seen.add(record["url"])
            self.stats["partial" if partial else "saved"] += 1
            if partial:
                self._check_circuit_breaker()   # only after the partial record is safely stored
        except ValidationError as exc:
            self.log.warning("Invalid record skipped: %s", exc)
            self.storage.log_failure(item.get("url", "?"), "validation", str(exc))
            self.stats["failed"] += 1
        except CircuitBreakerOpen:
            raise
        except Exception as exc:  # last-resort guard: one bad item never kills the run
            self.log.exception("Unexpected error on %s", item.get("url"))
            self.storage.log_failure(item.get("url", "?"), "unexpected", repr(exc))
            self.stats["failed"] += 1

    def _check_circuit_breaker(self):
        if self._consecutive_failures >= self.cfg.max_consecutive_failures:
            raise CircuitBreakerOpen(
                f"{self._consecutive_failures} consecutive failures - site may be blocking us")

    def _finalize(self):
        try:
            if self.records:
                paths = self.storage.export_all(self.records)
                self.log.info("Exported %d records -> %s", len(self.records),
                              ", ".join(str(p) for p in paths.values()))
            else:
                self.log.warning("No records collected; nothing to export")
        except OSError as exc:
            self.log.error("Export failed (data is still safe in %s): %s",
                           self.storage.jsonl, exc)
