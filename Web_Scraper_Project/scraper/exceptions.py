class ScraperError(Exception):
    """Base class for all scraper errors."""


#Network / HTTP layer
class FetchError(ScraperError):
    """A page could not be downloaded."""

    def __init__(self, url, message):
        super().__init__(f"{message} [{url}]")
        self.url = url


class PageNotFoundError(FetchError):
    """HTTP 404 / 410 - never retried."""


class HTTPStatusError(FetchError):
    """Non-retryable HTTP error such as 401/403/400."""


class RobotsDisallowedError(FetchError):
    """robots.txt forbids fetching this URL."""


class RetryExhaustedError(FetchError):
    """All retry attempts failed (timeouts, 5xx, 429, connection resets)."""


class InvalidContentError(FetchError):
    """Response is empty, too large, or not HTML."""


#Parsing / data layer
class ParseError(ScraperError):
    """HTML could not be parsed into the expected structure."""


class StructureChangedError(ParseError):
    """Key selectors no longer match - the site layout probably changed."""


class ValidationError(ScraperError):
    """A scraped record failed validation."""


#Control flow
class CircuitBreakerOpen(ScraperError):
    """Too many consecutive failures - stop scraping to avoid hammering a site."""
