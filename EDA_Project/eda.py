#!/usr/bin/env python3
"""Exploratory Data Analysis for the scraped books dataset (Task 1 output).

Usage:  python eda.py --input ../web_scraper_project/output/books.csv --out eda_output
Produces: report.html (self-contained), figures/*.png, data_issues.csv, tests.csv,
          summary_stats.csv, cleaned_books.csv
"""
import argparse
import base64
import html
import io
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger("eda")
ALPHA = 0.05
REQUIRED = ["title", "url", "price_gbp", "rating", "in_stock"]
NUMERIC = ["price_gbp", "rating", "stock_count", "num_reviews"]
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "font.size": 9})

QUESTIONS = [
    "Q1. What does the dataset contain - how many rows, which variables, which types, how complete?",
    "Q2. How is price distributed? Is it symmetric or skewed, and are there extreme values?",
    "Q3. Are star ratings balanced, or do some ratings dominate?",
    "Q4. Do more expensive books get higher (or lower) ratings?",
    "Q5. Do categories differ in price, and are enough books per category to compare them?",
    "Q6. Is stock status related to price or rating?",
    "Q7. Do books with longer descriptions cost more or rate higher?",
    "Q8. What data problems must be fixed before modelling or reporting?",
]


def boxplot(ax, data, labels=None, **kw):
    """Version-proof boxplot (matplotlib renamed labels -> tick_labels in 3.9). Always vertical."""
    try:
        return ax.boxplot(data, tick_labels=labels, **kw)
    except TypeError:
        return ax.boxplot(data, labels=labels, **kw)


# =============================================================== loading
def load_data(path: Path):
    """Read + coerce types. Returns (df, coercion_notes). Exits with a clear message on fatal problems."""
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except FileNotFoundError:
        sys.exit(f"Input file not found: {path}\nRun the Task 1 scraper first, or use sample_data/books_SYNTHETIC.csv")
    except pd.errors.EmptyDataError:
        sys.exit(f"Input file is empty: {path}")
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        sys.exit(f"Input file is not a readable CSV: {exc}")
    if df.empty:
        sys.exit("Input file has headers but no rows.")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"Input is missing required columns: {missing}. Found: {list(df.columns)}")

    notes = []
    for col in NUMERIC:
        if col in df.columns:
            before = df[col].notna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            lost = before - df[col].notna().sum()
            if lost:
                notes.append((col, int(lost)))
    for col in ("in_stock", "detail_ok"):
        if col in df.columns and df[col].dtype != bool:
            m = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
            mapped = df[col].astype(str).str.strip().str.lower().map(m)
            notes_bad = int(mapped.isna().sum() - df[col].isna().sum())
            if notes_bad > 0:
                notes.append((col, notes_bad))
            df[col] = mapped.astype("boolean")
    if "scraped_at" in df.columns:
        df["scraped_at"] = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
    return df, notes


# ======================================================== data issues
def detect_issues(raw: pd.DataFrame, coercion_notes) -> pd.DataFrame:
    rows = []

    def add(sev, issue, count, action):
        if count and count > 0:
            rows.append({"severity": sev, "issue": issue, "rows_affected": int(count), "recommended_action": action})

    for col, n in coercion_notes:
        add("High", f"Non-numeric/invalid values in '{col}' (became missing)", n, "Inspect the scraper's parser for this field")
    add("High", "Duplicate rows (same URL)", raw.duplicated("url").sum(), "Drop duplicates, keep first (done in cleaned_books.csv)")
    dedup = raw.drop_duplicates("url")
    for c in REQUIRED:
        add("High", f"Missing required value: {c}", raw[c].isna().sum(), "Drop or re-scrape those rows")
    add("High", "Price <= 0", (raw["price_gbp"] <= 0).sum(), "Treat as invalid; drop (done in cleaned file)")
    add("High", "Rating outside 1-5", (~raw["rating"].dropna().between(1, 5)).sum(), "Treat as invalid; set to missing")

    p = dedup["price_gbp"].where(dedup["price_gbp"] > 0).dropna()
    if len(p) >= 8:
        q1, q3 = p.quantile([.25, .75]); iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        add("Medium", f"Price outliers by IQR rule (outside {lo:.2f} to {hi:.2f})", ((p < lo) | (p > hi)).sum(),
            "Verify against the live page; keep if real, otherwise correct. Flagged, not removed")

    if "detail_ok" in raw.columns:
        add("Medium", "Detail page failed during scrape (optional fields missing)", (raw["detail_ok"] == False).sum(),  # noqa: E712
            "Re-run scraper with --resume or retry failed URLs from failures.csv")
    for c in ("upc", "category", "stock_count", "description", "num_reviews"):
        if c in raw.columns:
            add("Medium" if c in ("category", "upc") else "Low", f"Missing values in '{c}'", raw[c].isna().sum(),
                "Impute/exclude per analysis; check if linked to failed detail pages")
    if {"in_stock", "stock_count"} <= set(raw.columns):
        add("Medium", "Marked out of stock but stock_count > 0", ((raw["in_stock"] == False) & (raw["stock_count"] > 0)).sum(),  # noqa: E712
            "Check the availability parser; the two fields contradict each other")
        add("Medium", "Marked in stock but stock_count is 0", ((raw["in_stock"] == True) & (raw["stock_count"] == 0)).sum(),  # noqa: E712
            "Check the availability parser")
    if "category" in raw.columns:
        cat = raw["category"].dropna().astype(str)
        norm = cat.str.strip().str.lower()
        variants = cat.groupby(norm).nunique()
        bad = variants[variants > 1].index
        add("Medium", "Category spelling/whitespace/case variants", norm.isin(bad).sum(), "Strip whitespace and merge variants into the most common spelling (done in cleaned file)")
        small = dedup["category"].value_counts()
        add("Info", "Categories with fewer than 5 books (too small for group comparisons)", (small < 5).sum(),
            "Merge into 'Other' or exclude from category tests (this counts categories, not rows)")
    for c in raw.columns:
        if raw[c].nunique(dropna=True) <= 1 and raw[c].notna().any():
            add("Low", f"Constant column '{c}' (no information)", len(raw), "Exclude from modelling")
    if "in_stock" in dedup.columns and dedup["in_stock"].notna().any():
        minority = dedup["in_stock"].value_counts(normalize=True).min()
        if minority < .05:
            add("Info", f"Severe class imbalance in in_stock (minority {minority:.1%})", int(round(minority * len(dedup))),
                "Tests on this split have low power; use caution")
    ts = pd.to_datetime(raw["scraped_at"], errors="coerce", utc=True) if "scraped_at" in raw.columns else pd.Series(dtype="datetime64[ns, UTC]")
    if ts.notna().any():
        span = ts.max() - ts.min()
        if span > pd.Timedelta(hours=24):
            add("Info", f"Data collected over {span.days}+ days; prices/stock may have changed mid-collection", len(raw), "Record as a limitation")

    out = pd.DataFrame(rows, columns=["severity", "issue", "rows_affected", "recommended_action"])
    order = {"High": 0, "Medium": 1, "Low": 2, "Info": 3}
    return out.sort_values("severity", key=lambda s: s.map(order), kind="stable").reset_index(drop=True)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.drop_duplicates("url", keep="first").copy()
    df = df.dropna(subset=["title", "price_gbp"])
    df = df[df["price_gbp"] > 0]
    df.loc[~df["rating"].between(1, 5), "rating"] = np.nan
    if "category" in df.columns:
        cat = df["category"].astype("string").str.strip()
        key = cat.str.lower()
        canonical = cat.groupby(key).agg(lambda x: x.value_counts().idxmax())   # most common spelling wins
        df["category"] = key.map(canonical)
    df["desc_len"] = df["description"].astype("string").str.len() if "description" in df.columns else np.nan
    q1, q3 = df["price_gbp"].quantile([.25, .75]); iqr = q3 - q1
    df["price_outlier"] = (df["price_gbp"] < q1 - 1.5 * iqr) | (df["price_gbp"] > q3 + 1.5 * iqr)
    return df.reset_index(drop=True)


# ========================================================= statistics
def label_effect(kind, v):
    a = abs(v)
    cuts = {"eps2": (.01, .06, .14), "corr": (.1, .3, .5), "cramer": (.1, .3, .5), "w": (.1, .3, .5)}[kind]
    return ("negligible", "small", "medium", "large")[sum(a >= c for c in cuts)]


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[idx]))
        adj[idx] = running
    return adj


def run_tests(df: pd.DataFrame) -> pd.DataFrame:
    res = []

    def record(name, h0, test, stat, p, eff_kind, eff, n, note=""):
        res.append(dict(name=name, H0=h0, test=test, statistic=stat, p_value=p, effect_kind=eff_kind,
                        effect=eff, effect_size=label_effect(eff_kind, eff), n=n, note=note))

    def skipped(name, h0, why):
        res.append(dict(name=name, H0=h0, test="-", statistic=np.nan, p_value=np.nan, effect_kind="",
                        effect=np.nan, effect_size="", n=0, note="SKIPPED: " + why))

    def safe(name, h0, fn):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fn()
        except Exception as exc:  # a failing test must not kill the report
            log.warning("Test '%s' failed: %s", name, exc)
            skipped(name, h0, f"error ({type(exc).__name__})")

    # T1 price ~ rating
    h0 = "Price distribution is the same for every rating"
    def t1():
        d = df.dropna(subset=["price_gbp", "rating"])
        groups = [g["price_gbp"].values for _, g in d.groupby("rating") if len(g) >= 3]
        if len(groups) < 2:
            return skipped("Price differs by rating (Kruskal-Wallis)", h0, "fewer than 2 rating groups with n>=3")
        H, p = stats.kruskal(*groups)
        n, k = sum(map(len, groups)), len(groups)
        record("Price differs by rating (Kruskal-Wallis)", h0, "Kruskal-Wallis", H, p, "eps2", max(0, (H - k + 1) / (n - k)), n)
    safe("Price differs by rating (Kruskal-Wallis)", h0, t1)

    # T2 price vs rating monotonic
    h0 = "No monotonic association between price and rating (rho = 0)"
    def t2():
        d = df.dropna(subset=["price_gbp", "rating"])
        if len(d) < 10 or d["rating"].nunique() < 2:
            return skipped("Price vs rating association (Spearman)", h0, "need >=10 rows and varying ratings")
        r, p = stats.spearmanr(d["price_gbp"], d["rating"])
        record("Price vs rating association (Spearman)", h0, "Spearman rho", r, p, "corr", r, len(d))
    safe("Price vs rating association (Spearman)", h0, t2)

    # T3 price ~ category
    h0 = "Price distribution is the same in every category"
    def t3():
        if "category" not in df:
            return skipped("Price differs by category (Kruskal-Wallis)", h0, "no category column")
        d = df.dropna(subset=["category", "price_gbp"])
        groups = [g["price_gbp"].values for _, g in d.groupby("category") if len(g) >= 5]
        if len(groups) < 2:
            return skipped("Price differs by category (Kruskal-Wallis)", h0, "fewer than 2 categories with n>=5")
        H, p = stats.kruskal(*groups)
        n, k = sum(map(len, groups)), len(groups)
        record("Price differs by category (Kruskal-Wallis)", h0, f"Kruskal-Wallis ({k} categories, n>=5)", H, p, "eps2", max(0, (H - k + 1) / (n - k)), n)
    safe("Price differs by category (Kruskal-Wallis)", h0, t3)

    # T4 price ~ in_stock
    h0 = "Price distribution is the same for in-stock and out-of-stock books"
    def t4():
        d = df.dropna(subset=["price_gbp", "in_stock"])
        a, b = d.loc[d["in_stock"].astype(bool), "price_gbp"], d.loc[~d["in_stock"].astype(bool), "price_gbp"]
        if min(len(a), len(b)) < 5:
            return skipped("Price differs by stock status (Mann-Whitney U)", h0, f"smaller group has only {min(len(a), len(b))} rows")
        U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rb = 1 - 2 * U / (len(a) * len(b))            # rank-biserial correlation
        record("Price differs by stock status (Mann-Whitney U)", h0, "Mann-Whitney U", U, p, "corr", rb, len(d))
    safe("Price differs by stock status (Mann-Whitney U)", h0, t4)

    # T5 rating x in_stock
    h0 = "Rating and stock status are independent"
    def t5():
        d = df.dropna(subset=["rating", "in_stock"])
        ct = pd.crosstab(d["rating"], d["in_stock"])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return skipped("Rating vs stock status (chi-square)", h0, "contingency table smaller than 2x2")
        chi2, p, _, expected = stats.chi2_contingency(ct)
        v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        note = "Expected counts < 5 in some cells - chi-square approximation unreliable" if (expected < 5).any() else ""
        record("Rating vs stock status (chi-square)", h0, "Chi-square independence", chi2, p, "cramer", v, int(ct.values.sum()), note)
    safe("Rating vs stock status (chi-square)", h0, t5)

    # T6 description length ~ price
    h0 = "No monotonic association between description length and price"
    def t6():
        d = df.dropna(subset=["desc_len", "price_gbp"])
        if len(d) < 10 or d["desc_len"].nunique() < 2:
            return skipped("Description length vs price (Spearman)", h0, "need >=10 rows with descriptions")
        r, p = stats.spearmanr(d["desc_len"], d["price_gbp"])
        record("Description length vs price (Spearman)", h0, "Spearman rho", r, p, "corr", r, len(d))
    safe("Description length vs price (Spearman)", h0, t6)

    # T7 rating uniformity
    h0 = "All five ratings are equally common"
    def t7():
        counts = df["rating"].dropna().value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
        n = counts.sum()
        if n < 25:
            return skipped("Ratings are balanced (chi-square goodness-of-fit)", h0, "need >=25 rated books")
        chi2, p = stats.chisquare(counts.values)
        record("Ratings are balanced (chi-square goodness-of-fit)", h0, "Chi-square GOF vs uniform", chi2, p, "w", np.sqrt(chi2 / n), int(n))
    safe("Ratings are balanced (chi-square goodness-of-fit)", h0, t7)

    out = pd.DataFrame(res)
    ok = out["p_value"].notna()
    out["p_holm"] = np.nan
    if ok.any():
        out.loc[ok, "p_holm"] = holm(out.loc[ok, "p_value"].values)

    def verdict(r):
        if pd.isna(r["p_value"]):
            return "n/a"
        if r["p_holm"] < ALPHA:
            tail = " - but the effect is negligible, so practical importance is doubtful" if r["effect_size"] == "negligible" else ""
            return f"Reject H0 ({r['effect_size']} effect){tail}"
        return "Fail to reject H0 (no evidence of an effect)"
    out["verdict"] = out.apply(verdict, axis=1)
    return out


# ============================================================= report
class Report:
    def __init__(self, figdir: Path):
        self.parts, self.figdir, self.toc = [], figdir, []
        figdir.mkdir(parents=True, exist_ok=True)

    def h2(self, title):
        sid = f"s{len(self.toc) + 1}"
        self.toc.append((sid, title))
        self.parts.append(f'<h2 id="{sid}">{html.escape(title)}</h2>')

    def p(self, text, cls=""):  # text may contain trusted HTML built in this file
        self.parts.append(f'<p class="{cls}">{text}</p>')

    def table(self, df, index=False, float_format=None):
        self.parts.append(df.to_html(classes="t", border=0, index=index, float_format=float_format, na_rep="-"))

    def fig(self, fig, name, caption):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        (self.figdir / f"{name}.png").write_bytes(buf.getvalue())
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode()
        self.parts.append(f'<figure><img src="data:image/png;base64,{b64}" alt="{html.escape(caption)}">'
                          f'<figcaption>{html.escape(caption)}</figcaption></figure>')

    def safe_plot(self, name, caption, fn):
        try:
            self.fig(fn(), name, caption)
        except Exception as exc:
            log.warning("Plot '%s' failed: %s", name, exc)
            self.p(f"<em>Chart '{html.escape(name)}' could not be drawn ({html.escape(type(exc).__name__)}).</em>", "warn")

    def render(self, title):
        toc = "".join(f'<li><a href="#{i}">{html.escape(t)}</a></li>' for i, t in self.toc)
        return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
:root{{--bg:#fff;--fg:#1f2328;--mut:#59636e;--line:#d1d9e0;--soft:#f6f8fa;--warn:#9a6700;--bad:#cf222e}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0d1117;--fg:#e6edf3;--mut:#9198a1;--line:#3d444d;--soft:#161b22;--warn:#d29922;--bad:#f85149}}}}
body{{font:15px/1.6 system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;background:var(--bg);color:var(--fg)}}
h1{{font-size:1.7rem}}h2{{margin-top:2.4rem;border-bottom:1px solid var(--line);padding-bottom:.3rem}}
.t{{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto;margin:.8rem 0}}
.t th,.t td{{padding:5px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}.t th{{background:var(--soft)}}
figure{{margin:1rem 0}}img{{max-width:100%;background:#fff;border-radius:6px}}figcaption{{color:var(--mut);font-size:13px}}
.note{{background:var(--soft);border-left:4px solid var(--line);padding:.6rem 1rem;border-radius:4px}}
.warn{{border-left-color:var(--warn)}}.q{{color:var(--mut)}}code{{background:var(--soft);padding:1px 5px;border-radius:4px}}
</style></head><body><h1>{html.escape(title)}</h1>
<nav><strong>Contents</strong><ol>{toc}</ol></nav>{''.join(self.parts)}</body></html>"""


def fmt(x, nd=3):
    return "n/a" if pd.isna(x) else f"{x:.{nd}g}"


def build_report(raw, df, issues, tests, outdir: Path, source_name: str, synthetic: bool):
    R = Report(outdir / "figures")
    title = "Exploratory Data Analysis - Book Catalogue"
    if synthetic:
        R.parts.append('<p class="note warn"><strong>SYNTHETIC DATA.</strong> This report was generated from a fake dataset '
                       'with planted problems to test the pipeline. None of its findings describe the real website. '
                       'Run <code>eda.py</code> on your own scraped <code>books.csv</code> for real results.</p>')
    R.p(f"Source file: <code>{html.escape(source_name)}</code> | rows analysed after cleaning: <strong>{len(df)}</strong> "
        f"(of {len(raw)} raw) | significance level alpha = {ALPHA}, Holm-corrected across all tests.", "note")

    # ---- 0 questions
    R.h2("Questions we want to answer")
    R.parts.append("<ol class='q'>" + "".join(f"<li>{html.escape(q[4:])}</li>" for q in QUESTIONS) + "</ol>")

    # ---- 1 structure
    R.h2("Q1. Data structure and completeness")
    info = pd.DataFrame({"dtype": raw.dtypes.astype(str), "non_null": raw.notna().sum(), "missing_%": raw.isna().mean() * 100,
                         "unique": raw.nunique(), "example": [str(raw[c].dropna().iloc[0])[:40] if raw[c].notna().any() else "-" for c in raw.columns]})
    R.table(info.reset_index(names="column"), float_format="{:.1f}".format)
    R.p(f"Raw shape: {raw.shape[0]} rows x {raw.shape[1]} columns. Each row is one book (unit of observation = product page).")

    def p_missing():
        m = raw.isna().mean().mul(100).sort_values()
        fig, ax = plt.subplots(figsize=(7, 3.2)); ax.barh(m.index, m.values, color="#d29922")
        ax.set_xlabel("% missing"); ax.set_xlim(0, max(5, m.max() * 1.1)); return fig
    R.safe_plot("missing_values", "Share of missing values per column (raw data)", p_missing)

    # ---- 2 price
    R.h2("Q2. Price distribution")
    pr = df["price_gbp"]
    desc = pr.describe(percentiles=[.05, .25, .5, .75, .95]).to_frame("price_gbp").T
    desc["skew"], desc["kurtosis"] = pr.skew(), pr.kurt()
    R.table(desc.reset_index(drop=True), float_format="{:.2f}".format)
    sh_p = np.nan
    if 3 <= len(pr) <= 5000 or len(pr) > 5000:
        sh_p = stats.shapiro(pr.sample(min(len(pr), 5000), random_state=0)).pvalue
    n_out = int(df["price_outlier"].sum())
    normal_txt = ("rejects normality" if sh_p < ALPHA else "does not reject normality") if not pd.isna(sh_p) else "was not run"
    R.p(f"Median £{pr.median():.2f}, mean £{pr.mean():.2f}, skewness {pr.skew():.2f}. Shapiro-Wilk {normal_txt} (p = {fmt(sh_p)}). "
        f"<strong>{n_out}</strong> price(s) are flagged as IQR outliers. Because normality is not guaranteed and outliers can exist, "
        "the hypothesis tests below are rank-based (non-parametric) and do not assume a normal distribution.")

    def p_price():
        inside = pr[~df.loc[pr.index, "price_outlier"]]          # readable view: exclude IQR outliers
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
        ax[0].hist(inside, bins=20, color="#4c78a8", edgecolor="white")
        ax[0].set_title(f"Histogram (excl. {len(pr) - len(inside)} outlier(s))"); ax[0].set_xlabel("£")
        boxplot(ax[1], pr, widths=.5); ax[1].set_title("Boxplot (all prices)"); ax[1].set_ylabel("£")
        stats.probplot(inside, dist="norm", plot=ax[2]); ax[2].set_title("Normal Q-Q (excl. outliers)")
        return fig
    R.safe_plot("price_distribution", "Price distribution. Histogram and Q-Q exclude IQR outliers for readability; the boxplot shows everything", p_price)

    # ---- 3 rating
    R.h2("Q3. Rating balance")
    rc = df["rating"].dropna().astype(int).value_counts().reindex(range(1, 6), fill_value=0)
    R.table(pd.DataFrame({"rating": rc.index, "books": rc.values}))
    def p_rating():
        fig, ax = plt.subplots(figsize=(5.5, 3.2)); ax.bar(rc.index, rc.values, color="#59a14f")
        ax.axhline(rc.sum() / 5, ls="--", color="gray", label="if perfectly balanced"); ax.legend(); ax.set_xlabel("stars"); ax.set_ylabel("books")
        return fig
    R.safe_plot("rating_counts", "Books per star rating", p_rating)

    # ---- 4 price vs rating
    R.h2("Q4. Price versus rating")
    def p_pr():
        d = df.dropna(subset=["rating"]); ks = sorted(d["rating"].unique())
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        boxplot(ax, [d.loc[d["rating"] == k, "price_gbp"] for k in ks], [int(k) for k in ks]); ax.set_xlabel("rating"); ax.set_ylabel("price £")
        return fig
    R.safe_plot("price_by_rating", "Price by star rating", p_pr)

    # ---- 5 categories
    R.h2("Q5. Categories")
    if "category" in df and df["category"].notna().any():
        vc = df["category"].value_counts()
        top = vc.head(15)
        def p_cat():
            fig, ax = plt.subplots(figsize=(7, 4)); ax.barh(top.index[::-1], top.values[::-1], color="#4c78a8"); ax.set_xlabel("books"); return fig
        R.safe_plot("category_counts", f"Books per category (top {len(top)} of {len(vc)})", p_cat)
        big = vc[vc >= 5].head(10).index
        def p_catp():
            fig, ax = plt.subplots(figsize=(8, 4.2))
            boxplot(ax, [df.loc[df["category"] == c, "price_gbp"] for c in big], list(big)); ax.set_ylabel("price £")
            ax.tick_params(axis="x", rotation=40)
            return fig
        if len(big):
            R.safe_plot("price_by_category", "Price by category (categories with at least 5 books)", p_catp)
        R.p(f"{len(vc)} distinct categories; {int((vc < 5).sum())} have fewer than 5 books and were excluded from the category test.")
    else:
        R.p("No usable category data.", "warn")

    # ---- 6/7 relationships
    R.h2("Q6-Q7. Relationships between variables")
    num = df[["price_gbp", "rating", "stock_count", "num_reviews", "desc_len"]].apply(pd.to_numeric, errors="coerce")
    num = num.loc[:, num.nunique() > 1]
    def p_corr():
        c = num.corr(method="spearman"); fig, ax = plt.subplots(figsize=(5.2, 4.2))
        im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1); ax.grid(False)
        ax.set_xticks(range(len(c))); ax.set_xticklabels(c.columns, rotation=40, ha="right"); ax.set_yticks(range(len(c))); ax.set_yticklabels(c.columns)
        for i in range(len(c)):
            for j in range(len(c)):
                ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, shrink=.8); return fig
    R.safe_plot("correlation", "Spearman correlation matrix (constant columns omitted)", p_corr)
    def p_sc():
        d = df.dropna(subset=["stock_count", "price_gbp"]); fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.scatter(d["stock_count"], d["price_gbp"], s=10, alpha=.4); ax.set_xlabel("stock_count"); ax.set_ylabel("price £"); return fig
    R.safe_plot("price_vs_stock", "Price vs units in stock", p_sc)

    # ---- hypothesis tests
    R.h2("Hypothesis tests")
    R.p("Each test states a null hypothesis H0. <strong>p (Holm)</strong> adjusts for running several tests at once. "
        "A tiny p-value alone is not enough - check the <strong>effect size</strong>, because with many rows even trivial differences become 'significant'.")
    show = tests[["name", "H0", "test", "n", "statistic", "p_value", "p_holm", "effect", "effect_size", "verdict", "note"]].copy()
    R.table(show, float_format="{:.4g}".format)
    sig = tests[tests["verdict"].str.startswith("Reject")]
    R.p(f"<strong>{len(sig)}</strong> of {int(tests['p_value'].notna().sum())} tests rejected H0 after correction." +
        ("" if len(sig) else " No statistically supported relationships were found - consistent with these variables being unrelated in this sample."), "note")
    R.p("Caveats: correlation is not causation; tests assume independent observations (books on one site are plausibly independent, "
        "but not guaranteed); assumptions of Kruskal-Wallis include similar distribution shapes across groups.")

    # ---- issues
    R.h2("Q8. Data issues to fix before further analysis")
    if issues.empty:
        R.p("No data issues detected.")
    else:
        R.table(issues)
    R.p("<code>cleaned_books.csv</code> applies the safe fixes (de-duplication, invalid price/rating removal, category normalisation) and "
        "adds <code>desc_len</code> and <code>price_outlier</code>. Outliers are flagged, not deleted - decide after verifying them.")

    # ---- summary
    R.h2("Summary and next steps")
    items = [f"Dataset: {len(df)} clean books across {df['category'].nunique() if 'category' in df else 'n/a'} categories.",
             f"Price is {'right-skewed' if pr.skew() > .5 else 'left-skewed' if pr.skew() < -.5 else 'roughly symmetric'} (skew {pr.skew():.2f}), median £{pr.median():.2f}."]
    high = issues[issues["severity"] == "High"] if not issues.empty else issues
    items.append(f"{len(high)} high-severity data issue type(s) found." if len(high) else "No high-severity data issues.")
    items.append(f"{len(sig)} hypothesis test(s) significant after multiple-testing correction.")
    R.parts.append("<ul>" + "".join(f"<li>{html.escape(i)}</li>" for i in items) + "</ul>")
    R.parts.append("<p><strong>Next:</strong> fix the high/medium issues at the scraper level, re-scrape failed pages, then move to "
                   "modelling or dashboards (Task 3) using <code>cleaned_books.csv</code>.</p>")
    return R.render(title)


# ================================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=Path("../web_scraper_project/output/books.csv"))
    ap.add_argument("--out", type=Path, default=Path("eda_output"))
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    raw, notes = load_data(a.input)
    log.info("Loaded %d rows x %d cols", *raw.shape)
    issues = detect_issues(raw, notes)
    df = clean(raw)
    if len(df) < 10:
        log.warning("Only %d usable rows - statistics will be unreliable", len(df))
    tests = run_tests(df)

    try:
        a.out.mkdir(parents=True, exist_ok=True)
        synthetic = "synthetic" in a.input.name.lower()
        page = build_report(raw, df, issues, tests, a.out, a.input.name, synthetic)
        (a.out / "report.html").write_text(page, encoding="utf-8")
        issues.to_csv(a.out / "data_issues.csv", index=False)
        tests.drop(columns=["effect_kind"]).to_csv(a.out / "tests.csv", index=False)
        df.describe(include="all").T.to_csv(a.out / "summary_stats.csv")
        df.to_csv(a.out / "cleaned_books.csv", index=False, encoding="utf-8-sig")
    except OSError as exc:
        log.error("Could not write outputs: %s", exc)
        return 2
    log.info("Report: %s", (a.out / "report.html").resolve())
    log.info("%d issue types | %d/%d tests significant", len(issues),
             int(tests["verdict"].str.startswith("Reject").sum()), int(tests["p_value"].notna().sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
