from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin


@dataclass
class Config:
    # Target (books.toscrape.com is a sandbox site built for scraping practice)
    base_url: str = "https://books.toscrape.com/"
    start_path: str = "catalogue/page-1.html"

    # Identify yourself politely
    user_agent: str = "EducationalScraper/1.0 (contact: your-email@example.com)"

    # Network behaviour
    connect_timeout: float = 10.0
    read_timeout: float = 20.0
    max_retries: int = 4
    backoff_factor: float = 1.5      # sleep = factor * 2**(attempt-1)
    max_backoff: float = 60.0
    delay: float = 1.0               # polite delay between requests (seconds)
    max_response_bytes: int = 5 * 1024 * 1024
    respect_robots: bool = True

    # Crawl scope
    max_pages: int = 3
    fetch_details: bool = True       # visit each book's page for extra fields

    # Safety
    max_consecutive_failures: int = 5

    # Output
    output_dir: Path = Path("output")
    resume: bool = False

    @property
    def start_url(self) -> str:
        return urljoin(self.base_url, self.start_path)
