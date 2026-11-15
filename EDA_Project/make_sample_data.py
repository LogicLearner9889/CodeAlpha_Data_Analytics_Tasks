"""Generate a SYNTHETIC dataset with the same schema as Task 1's books.csv.

It exists only so the EDA pipeline can be tested without internet access. It contains deliberately
planted problems (duplicates, outliers, missing details, inconsistent stock...) that the EDA must find.
Do NOT draw real-world conclusions from it - run eda.py on your own scraped output/books.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make(n=1000, seed=42, inject_issues=True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cats = ["Default", "Nonfiction", "Sequential Art", "Add a comment", "Fiction", "Romance", "Mystery",
            "Historical Fiction", "Childrens", "Poetry", "Travel", "Science", "Humor", "Music", "Art"]
    w = np.array([8, 13, 7, 6, 12, 5, 5, 4, 5, 2, 4, 2, 2, 1, 1], float)
    category = rng.choice(cats, n, p=w / w.sum())
    price = np.round(rng.uniform(10, 60, n), 2)
    rating = rng.integers(1, 6, n)
    in_stock = rng.random(n) < 0.97
    stock = np.where(in_stock, rng.integers(1, 23, n), 0).astype(float)
    desc = [("Lorem ipsum " * int(rng.integers(5, 60))).strip() for _ in range(n)]
    df = pd.DataFrame({
        "title": [f"Book Title {i:04d}" for i in range(n)],
        "url": [f"https://books.toscrape.com/catalogue/book-{i:04d}/index.html" for i in range(n)],
        "upc": [f"{rng.integers(10**15, 10**16):x}"[:16] for _ in range(n)],
        "category": category, "price_gbp": price, "rating": rating, "in_stock": in_stock,
        "stock_count": stock, "num_reviews": 0, "description": desc, "detail_ok": True,
        "scraped_at": "2026-11-15T07:00:00+00:00",
    })
    if inject_issues:
        miss = rng.choice(n, 60, replace=False)                   # detail pages that failed
        df.loc[miss, ["upc", "category", "stock_count", "description"]] = np.nan
        df.loc[miss, "detail_ok"] = False
        df.loc[rng.choice(n, 40, replace=False), "description"] = np.nan   # genuinely no description
        df.loc[5, "price_gbp"] = 999.99                           # extreme outlier
        df.loc[6, "price_gbp"] = 0.0                              # impossible price
        df.loc[7, "rating"] = 9                                   # invalid rating
        df.loc[8, "category"] = " poetry "                        # inconsistent spelling
        df.loc[9, "category"] = "Poetry"
        df.loc[[10, 11, 12], "in_stock"] = False                  # out of stock but count > 0
        df.loc[[10, 11, 12], "stock_count"] = 5
        df = pd.concat([df, df.sample(12, random_state=1)], ignore_index=True)   # duplicate rows
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sample_data/books_SYNTHETIC.csv")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    make().to_csv(a.out, index=False, encoding="utf-8-sig")
    print("wrote", a.out)
