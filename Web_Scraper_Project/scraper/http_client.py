import logging
import time
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from requests.exceptions import (
    ChunkedEncodingError, ConnectionError as ReqConnectionError, InvalidURL,
    MissingSchema, RequestException, Timeout, TooManyRedirects,
)

from .config import Config
from .exceptions import (
    FetchError, HTTPStatusError, InvalidContentError, PageNotFoundError,
    RetryExhaustedError, RobotsDisallowedError,
)

RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


class _Retry(Exception):
    """Internal signal to jump to the retry/backoff step."""


class HttpClient:
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.log = logger
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": cfg.user_agent,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        })
        self._last_request = 0.0
        self._robots = {}  # origin -> RobotFileParser | "allow_all" | "deny_all"

    #robots.txt
    def _load_robots(self, origin: str):
        robots_url = f"{origin}/robots.txt"
        try:
            resp = self.session.get(
                robots_url, timeout=(self.cfg.connect_timeout, self.cfg.read_timeout)
            )
        except RequestException as exc:
            self.log.warning("Could not fetch robots.txt (%s); proceeding cautiously", exc)
            return "allow_all"
        if resp.status_code in (401, 403):
            self.log.warning("robots.txt returned %s -> treating site as disallowed", resp.status_code)
            return "deny_all"
        if resp.status_code >= 400:
            return "allow_all"  # no robots.txt = no restrictions
        parser = RobotFileParser()
        parser.parse(resp.text.splitlines())
        return parser

    def can_fetch(self, url: str) -> bool:
        parts = urlparse(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            self._robots[origin] = self._load_robots(origin)
        rules = self._robots[origin]
        if rules == "allow_all":
            return True
        if rules == "deny_all":
            return False
        return rules.can_fetch(self.cfg.user_agent, url)

    #throttling
    def _throttle(self):
        wait = self.cfg.delay - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _backoff(self, attempt: int, retry_after=None):
        delay = self.cfg.backoff_factor * (2 ** (attempt - 1))
        if retry_after is not None:
            delay = max(delay, retry_after)
        delay = min(delay, self.cfg.max_backoff)
        self.log.debug("Backing off %.2fs before retry", delay)
        time.sleep(delay)

    @staticmethod
    def _parse_retry_after(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    #GET
    def get(self, url: str) -> str:
        """Return decoded HTML or raise a FetchError subclass."""
        if self.cfg.respect_robots and not self.can_fetch(url):
            raise RobotsDisallowedError(url, "Blocked by robots.txt")

        last_problem = "unknown error"
        for attempt in range(1, self.cfg.max_retries + 1):
            self._throttle()
            retry_after = None
            try:
                with self.session.get(
                    url, timeout=(self.cfg.connect_timeout, self.cfg.read_timeout), stream=True
                ) as resp:
                    status = resp.status_code

                    if status in (404, 410):
                        raise PageNotFoundError(url, f"HTTP {status} Not Found")
                    if status in RETRY_STATUS:
                        retry_after = self._parse_retry_after(resp.headers.get("Retry-After"))
                        last_problem = f"HTTP {status}"
                        self.log.warning("Attempt %d/%d: %s for %s",
                                         attempt, self.cfg.max_retries, last_problem, url)
                        raise _Retry()
                    if status >= 400:
                        raise HTTPStatusError(url, f"HTTP {status}")

                    ctype = resp.headers.get("Content-Type", "")
                    if "html" not in ctype.lower() and "xml" not in ctype.lower():
                        raise InvalidContentError(url, f"Unexpected content type '{ctype}'")

                    body = self._read_limited(resp, url)
                    if not body.strip():
                        raise InvalidContentError(url, "Empty response body")
                    return self._decode(body, resp.encoding)

            except _Retry:
                pass
            except FetchError:
                raise
            except (Timeout, ReqConnectionError, ChunkedEncodingError) as exc:
                last_problem = f"{type(exc).__name__}: {exc}"
                self.log.warning("Attempt %d/%d failed: %s", attempt, self.cfg.max_retries, last_problem)
            except (InvalidURL, MissingSchema, TooManyRedirects) as exc:
                raise FetchError(url, f"Bad URL or redirect loop: {exc}") from exc
            except RequestException as exc:
                raise FetchError(url, f"Request failed: {exc}") from exc

            if attempt < self.cfg.max_retries:
                self._backoff(attempt, retry_after)

        raise RetryExhaustedError(
            url, f"Gave up after {self.cfg.max_retries} attempts ({last_problem})"
        )

    def _read_limited(self, resp, url: str) -> bytes:
        chunks, size = [], 0
        for chunk in resp.iter_content(8192):
            size += len(chunk)
            if size > self.cfg.max_response_bytes:
                raise InvalidContentError(url, "Response exceeds size limit")
            chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    def _decode(body: bytes, declared) -> str:
        # requests defaults to ISO-8859-1 for text/* with no charset, which garbles £ etc.
        for enc in ("utf-8", declared, "latin-1"):
            if not enc:
                continue
            try:
                return body.decode(enc)
            except (UnicodeDecodeError, LookupError):
                continue
        return body.decode("utf-8", errors="replace")
