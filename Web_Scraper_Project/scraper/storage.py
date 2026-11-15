import csv
import json
import os
import sqlite3
import tempfile
from pathlib import Path

FIELDS = ["title", "url", "upc", "category", "price_gbp", "rating", "in_stock",
          "stock_count", "num_reviews", "description", "detail_ok", "scraped_at"]


class Storage:
    def __init__(self, output_dir: Path):
        self.dir = Path(output_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.dir / "books.jsonl"
        self.failures = self.dir / "failures.csv"
        self.checkpoint = self.dir / "checkpoint.json"

    #incremental
    def reset(self):
        for p in (self.jsonl, self.failures, self.checkpoint):
            p.unlink(missing_ok=True)

    def append(self, record: dict):
        with open(self.jsonl, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def load_records(self) -> list:
        records = []
        if not self.jsonl.exists():
            return records
        with open(self.jsonl, encoding="utf-8") as fh:
            for line in fh:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a half-written last line after a crash is ignored
        return records

    def log_failure(self, url: str, stage: str, error: str):
        new = not self.failures.exists()
        with open(self.failures, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["url", "stage", "error"])
            w.writerow([url, stage, error])

    def save_checkpoint(self, next_url, pages_done: int):
        self._atomic_write(
            self.checkpoint,
            lambda fh: json.dump({"next_url": next_url, "pages_done": pages_done}, fh),
            mode="w", encoding="utf-8")

    def load_checkpoint(self):
        try:
            return json.loads(self.checkpoint.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    # ---- final exports ----------------------------------------------------
    def export_all(self, records: list) -> dict:
        paths = {"csv": self.dir / "books.csv", "json": self.dir / "books.json",
                 "sqlite": self.dir / "books.db"}

        def write_csv(fh):
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(records)

        self._atomic_write(paths["csv"], write_csv, mode="w", newline="", encoding="utf-8-sig")
        self._atomic_write(paths["json"],
                           lambda fh: json.dump(records, fh, indent=2, ensure_ascii=False),
                           mode="w", encoding="utf-8")
        self._export_sqlite(paths["sqlite"], records)
        return paths

    def _export_sqlite(self, path: Path, records: list):
        con = sqlite3.connect(path)
        try:
            with con:
                con.execute("""CREATE TABLE IF NOT EXISTS books (
                    url TEXT PRIMARY KEY, title TEXT NOT NULL, upc TEXT, category TEXT,
                    price_gbp REAL, rating INTEGER, in_stock INTEGER, stock_count INTEGER,
                    num_reviews INTEGER, description TEXT, detail_ok INTEGER, scraped_at TEXT)""")
                cols = ["url", "title", "upc", "category", "price_gbp", "rating", "in_stock",
                        "stock_count", "num_reviews", "description", "detail_ok", "scraped_at"]
                con.executemany(
                    "INSERT OR REPLACE INTO books VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    [tuple(r.get(c) for c in cols) for r in records])
        finally:
            con.close()

    @staticmethod
    def _atomic_write(path: Path, writer, mode="w", **kwargs):
        """Write to a temp file then rename, so a crash never leaves a corrupt file."""
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, mode, **kwargs) as fh:
                writer(fh)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
