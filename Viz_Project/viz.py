"""
Usage:
python viz.py --input ../Eda_Project/eda_output/cleaned_books.csv --raw ../Web_Scraper_Project/output/books.csv
"""
import argparse
import base64
import html
import logging
import sys
import textwrap
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy import stats

log = logging.getLogger("viz")
ALPHA = 0.05

# Okabe-Ito based, colour-blind safe. ONE accent for emphasis, grey for context, vermillion only to flag.
PAL = dict(accent="#0072B2", flag="#D55E00", gray="#9AA0A6", light="#DDE1E5", ink="#1F2328", mute="#59636E", tile="#F4F6F8")


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": PAL["light"], "axes.labelcolor": PAL["mute"], "xtick.color": PAL["mute"],
        "ytick.color": PAL["mute"], "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#EEF0F2", "axes.axisbelow": True,
    })


#loading
def load(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except FileNotFoundError:
        sys.exit(f"Input not found: {path}\nRun Task 1 (scraper) and Task 2 (EDA) first, or use the synthetic demo data.")
    except pd.errors.EmptyDataError:
        sys.exit(f"Input is empty: {path}")
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        sys.exit(f"Input is not a readable CSV: {exc}")
    missing = [c for c in ("price_gbp", "rating") if c not in df.columns]
    if missing:
        sys.exit(f"Input is missing required columns {missing}. Found: {list(df.columns)}")
    for c in ("price_gbp", "rating", "stock_count", "desc_len"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "in_stock" in df.columns:
        df["in_stock"] = df["in_stock"].astype(str).str.strip().str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}).astype("boolean")
    df = df.dropna(subset=["price_gbp"])
    df = df[df["price_gbp"] > 0].reset_index(drop=True)
    if df.empty:
        sys.exit("No rows with a valid price - nothing to visualise.")
    return df


#statistics
def boot_median_ci(x, n_boot=1000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(n_boot, len(x))), axis=1)
    return tuple(np.percentile(meds, [2.5, 97.5]))


def holm(pvals):
    p = np.asarray(pvals, float); m = len(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (m - rank) * p[i])); adj[i] = run
    return adj


def price_bands(inside: pd.Series, k=5):
    if inside.empty or inside.max() <= inside.min():          # all prices identical: a single band
        label = f"₹{inside.min():.0f}" if len(inside) else "n/a"
        return pd.Series([label] * len(inside), index=inside.index, dtype="object"), [label]
    edges = np.linspace(inside.min(), inside.max(), k + 1)
    labels = [f"₹{edges[i]:.0f}–{edges[i + 1]:.0f}" for i in range(k)]
    return pd.cut(inside, bins=edges, include_lowest=True, labels=labels), labels


def compute_facts(df: pd.DataFrame, raw=None) -> dict:
    f = {"n": len(df)}
    p = df["price_gbp"]
    f["q1"], f["med"], f["q3"] = p.quantile([.25, .5, .75])
    iqr = f["q3"] - f["q1"]
    f["lo_fence"], f["hi_fence"] = f["q1"] - 1.5 * iqr, f["q3"] + 1.5 * iqr
    f["inside"] = p[(p >= f["lo_fence"]) & (p <= f["hi_fence"])]
    f["n_outliers"] = int(len(p) - len(f["inside"]))
    f["pmin"], f["pmax"] = f["inside"].min(), f["inside"].max()
    f["skew"], f["kurt"] = float(p.skew()), float(f["inside"].kurt()) if len(f["inside"]) > 3 else 0.0

    r = df["rating"].dropna()
    r = r[r.between(1, 5)].astype(int)
    f["rating_counts"] = r.value_counts().reindex(range(1, 6), fill_value=0)
    f["rating_mean"] = float(r.mean()) if len(r) else np.nan
    f["in_stock_pct"] = float(df["in_stock"].mean()) if "in_stock" in df and df["in_stock"].notna().any() else np.nan
    f["n_categories"] = int(df["category"].nunique()) if "category" in df else 0

    tests = {}# name -> (p, effect, kind)
    if len(r) >= 25:
        chi2, pv = stats.chisquare(f["rating_counts"].values)
        tests["rating"] = (pv, np.sqrt(chi2 / len(r)), "w")
    d = df.dropna(subset=["rating"]); d = d[d["rating"].between(1, 5)]
    f["rating_stats"] = None
    if len(d) >= 10 and d["rating"].nunique() > 1:
        rho, pv = stats.spearmanr(d["price_gbp"], d["rating"])
        tests["price_rating"] = (pv, rho, "rho")
        rows = []
        for k, g in d.groupby("rating"):
            lo, hi = boot_median_ci(g["price_gbp"], seed=int(k))
            rows.append((int(k), len(g), g["price_gbp"].median(), lo, hi))
        f["rating_stats"] = pd.DataFrame(rows, columns=["rating", "n", "median", "lo", "hi"])
    f["cat_stats"] = None
    if "category" in df and df["category"].notna().any():
        dc = df.dropna(subset=["category"])
        sizes = dc["category"].value_counts()
        min_n = 15 if (sizes >= 15).sum() >= 3 else 5
        keep = sizes[sizes >= min_n].index
        f["cat_min_n"] = min_n
        if len(keep) >= 2:
            groups = [dc.loc[dc["category"] == c, "price_gbp"].values for c in keep]
            H, pv = stats.kruskal(*groups)
            n_, k_ = sum(map(len, groups)), len(groups)
            tests["category"] = (pv, max(0.0, (H - k_ + 1) / (n_ - k_)), "eps2")
            rows = []
            for i, c in enumerate(keep):
                lo, hi = boot_median_ci(groups[i], seed=i)
                rows.append((c, len(groups[i]), np.median(groups[i]), lo, hi))
            f["cat_stats"] = pd.DataFrame(rows, columns=["category", "n", "median", "lo", "hi"]).sort_values("median").reset_index(drop=True)
    f["heat"] = None
    if len(f["inside"]) >= 30 and f["pmax"] > f["pmin"] and df["rating"].notna().sum() >= 30:   # constant prices cannot be binned
        band, labels = price_bands(f["inside"])
        tmp = pd.DataFrame({"band": band, "rating": df.loc[f["inside"].index, "rating"]}).dropna()
        tmp = tmp[tmp["rating"].between(1, 5)]
        mat = pd.crosstab(tmp["rating"].astype(int), tmp["band"]).reindex(index=range(1, 6), columns=labels, fill_value=0)
        f["heat"] = mat
        nz = mat.loc[mat.sum(axis=1) > 0, mat.sum(axis=0) > 0]
        if nz.shape[0] >= 2 and nz.shape[1] >= 2:
            chi2, pv, _, _ = stats.chi2_contingency(nz)
            tests["heat"] = (pv, np.sqrt(chi2 / (nz.values.sum() * (min(nz.shape) - 1))), "v")

    names = list(tests)
    adj = dict(zip(names, holm([tests[n][0] for n in names]))) if names else {}
    thresholds = {"w": .1, "rho": .1, "eps2": .01, "v": .1}
    f["tests"] = {n: dict(p=tests[n][0], p_holm=adj[n], effect=tests[n][1], kind=tests[n][2],
                          notable=bool(adj[n] < ALPHA and abs(tests[n][1]) >= thresholds[tests[n][2]])) for n in names}

    f["raw"] = None
    if raw is not None:
        n_raw = len(raw)
        dedup = raw.drop_duplicates("url") if "url" in raw else raw
        pr = pd.to_numeric(dedup.get("price_gbp"), errors="coerce")
        valid = dedup[pr > 0]
        incomplete = int((valid["detail_ok"].astype(str).str.lower() == "false").sum()) if "detail_ok" in valid else 0
        f["raw"] = dict(n_raw=n_raw, n_dedup=len(dedup), n_valid=len(valid), n_incomplete=incomplete,
                        n_complete=len(valid) - incomplete)
    return f


#insights (headline + so-what)
def insights(f: dict) -> dict:
    I, T = {}, f["tests"]
    gbp = lambda x: f"£{x:.0f}"

    flat = f["kurt"] < -1.0
    I["price"] = dict(
        headline=f"Half of all books cost between {gbp(f['q1'])} and {gbp(f['q3'])}",
        sub=f"Price distribution of {f['n']:,} books · median £{f['med']:.2f}" + (f" · {f['n_outliers']} extreme price(s) not shown" if f["n_outliers"] else ""),
        takeaway=(f"There is no single 'typical' price: books are spread almost evenly from about {gbp(f['pmin'])} to {gbp(f['pmax'])}. "
                  "Pricing rules should therefore use bands, not a single average.") if flat else
                 (f"The typical book costs about {gbp(f['med'])}; the distribution is "
                  f"{'right-skewed' if f['skew'] > .5 else 'left-skewed' if f['skew'] < -.5 else 'roughly symmetric'}, so the median describes it better than the mean."))

    if "rating" in T:
        t, top = T["rating"], int(f["rating_counts"].idxmax())
        share = f["rating_counts"].max() / f["rating_counts"].sum()
        if t["notable"]:
            head, take = f"{top}-star is the most common rating ({share:.0%} of books)", f"Ratings are unbalanced (chi-square p = {t['p']:.3g}); {top} stars dominates."
        else:
            head, take = "Ratings are spread evenly across 1 to 5 stars", (f"No rating dominates: the split is statistically indistinguishable from even (p = {t['p']:.2f}). "
                                                                           "A star rating alone therefore separates books very little.")
        I["rating"] = dict(headline=head, sub=f"Books per star rating (n = {int(f['rating_counts'].sum()):,}) · dashed line = perfectly even split", takeaway=take)

    if "price_rating" in T:
        t = T["price_rating"]
        if t["notable"]:
            head = f"Price {'rises' if t['effect'] > 0 else 'falls'} with rating (rank correlation ρ = {t['effect']:+.2f})"
            take = f"The association is statistically reliable (adjusted p = {t['p_holm']:.3g}); check whether the size matters for your decision."
        else:
            head = "Higher-rated books do not cost more"
            take = (f"Rank correlation between price and rating is ρ = {t['effect']:+.2f} (adjusted p = {t['p_holm']:.2f}): no reliable link. "
                    "Do not use price as a stand-in for quality.")
        I["price_rating"] = dict(headline=head, sub="Median price per rating with 95% bootstrap interval · dots = individual books", takeaway=take)

    if f["cat_stats"] is not None:
        cs, t = f["cat_stats"], T.get("category")
        if t and t["notable"]:
            hi_, lo_ = cs.iloc[-1], cs.iloc[0]
            head = f"{hi_['category']} is the priciest category (median {gbp(hi_['median'])}); {lo_['category']} the cheapest ({gbp(lo_['median'])})"
            take = f"Category differences are statistically reliable (adjusted p = {t['p_holm']:.3g}, effect size ε² = {t['effect']:.3f})."
        else:
            head = "Category barely changes price: the differences are within noise"
            take = (f"Median prices overlap across categories (Kruskal-Wallis adjusted p = {t['p_holm']:.2f}). " if t else "") + \
                   "A single price policy is defensible; category-specific rules are not supported by this data."
        I["category"] = dict(headline=head, sub=f"Median price by category (categories with ≥ {f['cat_min_n']} books) · bars = 95% bootstrap interval", takeaway=take)

    if f["heat"] is not None and "heat" in T:
        t = T["heat"]
        head = ("Price and rating are linked: some price bands hold far more of certain ratings" if t["notable"]
                else "No price–rating pattern: books fill every cell about equally")
        I["heat"] = dict(headline=head, sub="Number of books by rating and price band (extreme prices excluded)",
                         takeaway=f"Independence test: adjusted p = {t['p_holm']:.2f}, Cramér's V = {t['effect']:.2f}." + ("" if t["notable"] else " Nothing here suggests a segment worth targeting."))

    if f["raw"]:
        r = f["raw"]; lost = r["n_raw"] - r["n_valid"]
        take = (f"{r['n_incomplete']} usable books are missing detail-page fields (category, UPC, description). Re-scrape them before category-level reporting."
                if r["n_incomplete"] else "All usable books have complete detail fields.")
        I["quality"] = dict(headline=f"{lost} of {r['n_raw']:,} scraped rows were dropped before analysis" if lost else "Every scraped row was usable",
                            sub="From raw scrape to analysis-ready data", takeaway=take)
    return I


#chart drawers (axes level)
def money(ax, axis="x"):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"£{v:.0f}")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def draw_price(ax, df, f, compact=False):
    counts, edges = np.histogram(f["inside"], bins=20)
    centers = (edges[:-1] + edges[1:]) / 2
    cols = [PAL["accent"] if f["q1"] <= c <= f["q3"] else PAL["light"] for c in centers]
    ax.bar(centers, counts, width=np.diff(edges) * .92, color=cols)
    ax.axvline(f["med"], color=PAL["ink"], lw=1.4)
    top = counts.max()
    ax.set_ylim(0, top * 1.28)
    ax.text(f["med"] + .4, top * 1.17, f"Median £{f['med']:.0f}", color=PAL["ink"], fontsize=9, va="center")
    ax.text(f["med"] - .5, top * 1.17, "Middle 50% of books", color=PAL["accent"], fontsize=9, fontweight="bold", va="center", ha="right")
    ax.set_xlabel("Price"); ax.set_ylabel("Books"); money(ax)
    ax.grid(axis="x", visible=False)


def draw_rating(ax, df, f, compact=False):
    c = f["rating_counts"]
    bars = ax.bar(c.index, c.values, color=PAL["accent"], width=.62)
    even = c.sum() / 5
    ax.axhline(even, color=PAL["ink"], ls="--", lw=1)
    ax.text(5.55, even, "Even split", va="center", fontsize=9, color=PAL["ink"])
    for b, v in zip(bars, c.values):
        ax.text(b.get_x() + b.get_width() / 2, v * .93, f"{v / max(c.sum(), 1):.0%}", ha="center", va="top", fontsize=10, fontweight="bold", color="white")
    ax.set_xticks(range(1, 6)); ax.set_xticklabels([f"{i}★" for i in range(1, 6)])
    ax.set_xlim(.4, 6.4); ax.set_ylim(0, max(c.max(), even) * 1.18); ax.set_ylabel("Books")
    ax.grid(axis="x", visible=False)


def draw_price_rating(ax, df, f, compact=False):
    d = df.dropna(subset=["rating"]); d = d[d["rating"].between(1, 5)].copy(); d["rating"] = d["rating"].astype(int)
    d = d[(d["price_gbp"] >= f["lo_fence"]) & (d["price_gbp"] <= f["hi_fence"])]        # outliers would flatten the view
    samp = pd.concat([g.sample(min(len(g), 120), random_state=0) for _, g in d.groupby("rating")])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.stripplot(data=samp, x="rating", y="price_gbp", order=range(1, 6), color=PAL["gray"], alpha=.28, size=3, jitter=.28, ax=ax, zorder=1)
    ax.axhline(f["med"], color=PAL["ink"], ls="--", lw=1, zorder=2)
    ax.text(4.55, f["med"], f"Overall\nmedian £{f['med']:.0f}", va="center", ha="left", fontsize=9, color=PAL["ink"])
    ax.set_xlim(-.6, 5.5)
    pad = max((f["pmax"] - f["pmin"]) * .06, 1.0)
    ax.set_ylim(max(0, f["pmin"] - pad), f["pmax"] + pad)
    rs = f["rating_stats"]
    for _, r in rs.iterrows():
        x = int(r["rating"]) - 1
        ax.plot([x, x], [r["lo"], r["hi"]], color=PAL["accent"], lw=3, zorder=3, solid_capstyle="round")
        ax.scatter([x], [r["median"]], s=60, color=PAL["accent"], edgecolor="white", zorder=4)
    ax.set_xticks(range(5)); ax.set_xticklabels([f"{i}★" for i in range(1, 6)])
    ax.set_xlabel("Rating"); ax.set_ylabel("Price"); money(ax, "y"); ax.grid(axis="x", visible=False)


def draw_category(ax, df, f, compact=False):
    cs = f["cat_stats"]; t = f["tests"].get("category")
    flag = bool(t and t["notable"])
    ypos = np.arange(len(cs))
    for y, (_, r) in zip(ypos, cs.iterrows()):
        out = flag and (r["lo"] > f["med"] or r["hi"] < f["med"])
        col = PAL["flag"] if out else PAL["accent"]
        ax.plot([r["lo"], r["hi"]], [y, y], color=col, lw=3, solid_capstyle="round")
        ax.scatter([r["median"]], [y], s=55, color=col, edgecolor="white", zorder=3)
        ax.text(ax.get_xlim()[1] if False else r["hi"] + .6, y, f"n={int(r['n'])}", va="center", fontsize=8, color=PAL["mute"], bbox=dict(fc="white", ec="none", pad=.8), zorder=5)
    ax.axvline(f["med"], color=PAL["ink"], ls="--", lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([textwrap.shorten(c, 15, placeholder="…") if compact else c for c in cs["category"]]); money(ax)
    ax.set_xlabel("Median price"); ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    ax.margins(x=.12)


def draw_heat(ax, df, f, compact=False):
    m = f["heat"]
    sns.heatmap(m, annot=True, fmt="d", cmap=sns.light_palette(PAL["accent"], as_cmap=True), cbar=False, linewidths=1.5,
                linecolor="white", ax=ax, annot_kws={"fontsize": 9})
    ax.invert_yaxis(); ax.set_yticklabels([f"{i}★" for i in m.index], rotation=0)
    ax.set_xticklabels(m.columns, rotation=0, fontsize=8)
    ax.set_xlabel("Price band"); ax.set_ylabel("Rating"); ax.grid(False)


def draw_quality(ax, df, f, compact=False):
    r = f["raw"]
    names = ["Scraped", "No duplicates", "Valid price", "Full details"] if compact else \
            ["Scraped rows", "After removing duplicates", "After removing invalid prices", "With complete detail fields"]
    stages = list(zip(names, [r["n_raw"], r["n_dedup"], r["n_valid"], r["n_complete"]]))
    ypos = np.arange(len(stages))[::-1]
    for y, (lab, v), col in zip(ypos, stages, [PAL["light"]] * 3 + [PAL["accent"]]):
        ax.barh(y, v, color=col, height=.6)
        ax.text(v, y, f"  {v:,} ({v / max(r['n_raw'], 1):.0%})", va="center", fontsize=9, color=PAL["ink"])
    ax.set_yticks(ypos); ax.set_yticklabels([s[0] for s in stages]); ax.set_xlim(0, r["n_raw"] * 1.3)
    ax.set_xlabel("Rows"); ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)


CHARTS = [("price", draw_price, lambda f: True), ("rating", draw_rating, lambda f: "rating" in f["tests"]),
          ("price_rating", draw_price_rating, lambda f: f["rating_stats"] is not None),
          ("category", draw_category, lambda f: f["cat_stats"] is not None),
          ("heat", draw_heat, lambda f: f["heat"] is not None), ("quality", draw_quality, lambda f: f["raw"] is not None)]


#figure assembly
def standalone(draw, df, f, ins, footer, size=(9, 5.4)):
    left = .24 if draw is draw_category else .105
    fig, ax = plt.subplots(figsize=size)
    if draw is draw_quality:
        left = .27
    fig.subplots_adjust(top=.76, bottom=.15, left=left, right=.96)
    draw(ax, df, f)
    fig.text(.03, .965, textwrap.fill(ins["headline"], 58), fontsize=17, fontweight="bold", color=PAL["ink"], va="top")
    fig.text(.03, .855, textwrap.fill(ins["sub"], 100), fontsize=9.5, color=PAL["mute"], va="top")
    fig.text(.03, .02, footer, fontsize=8, color=PAL["mute"])
    return fig


def dashboard(df, f, I, footer):
    fig = plt.figure(figsize=(17, 10.6))
    gk = fig.add_gridspec(1, 6, left=.05, right=.98, top=.855, bottom=.76, wspace=.07)
    gc = fig.add_gridspec(2, 3, left=.05, right=.98, top=.665, bottom=.075, hspace=.62, wspace=.34)
    fig.text(.05, .965, "Book catalogue at a glance", fontsize=24, fontweight="bold", color=PAL["ink"], va="top")
    fig.text(.05, .915, "Price, rating and data-quality overview · each panel title states the finding", fontsize=11, color=PAL["mute"], va="top")
    kpis = [(f"{f['n']:,}", "books analysed"), (f"£{f['med']:.0f}", "median price"), (f"£{f['q1']:.0f}–{f['q3']:.0f}", "middle 50% of prices"),
            (f"{f['rating_mean']:.1f}★" if not np.isnan(f["rating_mean"]) else "n/a", "average rating"),
            (f"{f['in_stock_pct']:.0%}" if not np.isnan(f["in_stock_pct"]) else "n/a", "in stock"), (f"{f['n_categories']}", "categories")]
    for i, (val, lab) in enumerate(kpis):
        ax = fig.add_subplot(gk[0, i]); ax.axis("off")
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, color=PAL["tile"]))
        ax.text(.07, .6, val, fontsize=22, fontweight="bold", color=PAL["accent"], transform=ax.transAxes, va="center")
        ax.text(.07, .2, lab, fontsize=9.5, color=PAL["mute"], transform=ax.transAxes, va="center")
    keys = ["price", "rating", "price_rating", "category", "heat", "quality"]
    drawers = {k: (d, ok) for k, d, ok in CHARTS}
    for n, key in enumerate(keys):
        ax = fig.add_subplot(gc[n // 3, n % 3])
        draw, ok = drawers[key]
        if key in I and ok(f):
            try:
                draw(ax, df, f, compact=True)
                if key == "category":
                    pos = ax.get_position(); ax.set_position([pos.x0 + .035, pos.y0, pos.width - .035, pos.height])
                ax.set_title(textwrap.fill(I[key]["headline"], 44), loc="left", fontsize=10.5, fontweight="bold", color=PAL["ink"], pad=10)
                continue
            except Exception as exc:
                log.warning("Dashboard panel %s failed: %s", key, exc)
        ax.clear(); ax.axis("off")
        ax.text(.5, .5, f"{key.replace('_', ' ')}: not available\n(insufficient or missing data)", ha="center", va="center", color=PAL["mute"])
    fig.text(.05, .015, footer, fontsize=8.5, color=PAL["mute"])
    return fig


#outputs
def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def tableau_extract(df: pd.DataFrame, f: dict, out: Path):
    t = df.copy()
    band, _ = price_bands(f["inside"])
    t["price_band"] = band.reindex(t.index).astype("object").fillna("Outlier (excluded from bands)")
    t["rating_label"] = t["rating"].map(lambda v: f"{int(v)} stars" if pd.notna(v) and 1 <= v <= 5 else "Unrated")
    if "in_stock" in t:
        t["in_stock"] = t["in_stock"].map({True: "In stock", False: "Out of stock"}).fillna("Unknown")
    cols = [c for c in ["title", "category", "price_gbp", "price_band", "rating", "rating_label", "in_stock", "stock_count",
                        "desc_len", "price_outlier", "detail_ok", "url"] if c in t.columns]
    t[cols].to_csv(out, index=False, encoding="utf-8-sig")


STORY_CSS = """
:root{--bg:#fff;--fg:#1f2328;--mut:#59636e;--line:#d1d9e0;--soft:#f4f6f8;--acc:#0072b2;--warn:#9a6700}
@media(prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--mut:#9198a1;--line:#3d444d;--soft:#161b22;--acc:#56b4e9;--warn:#d29922}}
*{box-sizing:border-box}body{font:16px/1.65 Georgia,'Times New Roman',serif;max-width:860px;margin:0 auto;padding:2.5rem 1.2rem 4rem;background:var(--bg);color:var(--fg)}
h1{font:700 2rem/1.2 system-ui,sans-serif;margin:.2rem 0 .4rem}h2{font:700 1.25rem system-ui,sans-serif;margin:2.6rem 0 .6rem;color:var(--acc)}
.dek{color:var(--mut);font:1.05rem system-ui,sans-serif}.kicker{font:600 .8rem system-ui,sans-serif;letter-spacing:.08em;text-transform:uppercase;color:var(--mut)}
figure{margin:1.2rem 0}figure img{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fff}
.so{border-left:4px solid var(--acc);padding:.2rem 1rem;margin:.6rem 0 0;background:var(--soft);border-radius:0 6px 6px 0}
.so b{font-family:system-ui,sans-serif}.warn{border-left-color:var(--warn)}ul{padding-left:1.2rem}li{margin:.35rem 0}
footer{margin-top:3rem;color:var(--mut);font:.85rem system-ui,sans-serif;border-top:1px solid var(--line);padding-top:1rem}
"""


def build_story(f, I, charts_dir: Path, dash_path: Path, synthetic: bool, source: str) -> str:
    esc = html.escape
    parts = ['<p class="kicker">Data story</p><h1>What the book catalogue data does and doesn’t tell us</h1>']
    parts.append(f'<p class="dek">{f["n"]:,} books · median price £{f["med"]:.0f} · {f["n_categories"]} categories</p>')
    if synthetic:
        parts.append('<p class="so warn"><b>Synthetic demo data.</b> These charts were built from fake data to demonstrate the pipeline. '
                     'Run <code>viz.py</code> on your own scrape to get real findings.</p>')
    notable = [k for k, v in f["tests"].items() if v["notable"]]
    parts.append("<h2>The short version</h2><ul>")
    parts.append(f"<li>{esc(I['price']['headline'])}.</li>")
    for k in ("price_rating", "category", "rating"):
        if k in I:
            parts.append(f"<li>{esc(I[k]['headline'])}.</li>")
    if "quality" in I:
        parts.append(f"<li>{esc(I['quality']['headline'])}.</li>")
    parts.append("</ul>")
    parts.append(f'<figure><img alt="Dashboard summarising price, rating, category and data quality" src="data:image/png;base64,{b64(dash_path)}">'
                 '<figcaption class="dek" style="font-size:.85rem">The one-page dashboard. Each panel is explained below.</figcaption></figure>')
    parts.append("<h2>The evidence, one chart at a time</h2>")
    for i, key in enumerate(["price", "rating", "price_rating", "category", "heat", "quality"], 1):
        p = charts_dir / f"{i:02d}_{key}.png"
        if key in I and p.exists():
            alt = f"{I[key]['headline']}. {I[key]['sub']}"
            parts.append(f'<figure><img alt="{esc(alt)}" src="data:image/png;base64,{b64(p)}"></figure>'
                         f'<p class="so"><b>So what:</b> {esc(I[key]["takeaway"])}</p>')
    parts.append("<h2>What this means for decisions</h2><ul>")
    if "price_rating" in f["tests"] and not f["tests"]["price_rating"]["notable"]:
        parts.append("<li><b>Don’t price by rating.</b> No reliable link between the two, so rating-based price tiers are unsupported.</li>")
    elif "price_rating" in f["tests"]:
        parts.append("<li><b>Investigate the price–rating link.</b> It is reliable; confirm the effect is big enough to act on.</li>")
    if "category" in f["tests"] and not f["tests"]["category"]["notable"]:
        parts.append("<li><b>One pricing policy is enough.</b> Category does not meaningfully move price here.</li>")
    elif "category" in f["tests"]:
        parts.append("<li><b>Review category pricing.</b> Categories differ reliably; use the interval chart to see which ones.</li>")
    if f["raw"] and f["raw"]["n_incomplete"]:
        parts.append(f"<li><b>Fix the scrape first.</b> {f['raw']['n_incomplete']} books lack detail fields; re-scrape before category reporting.</li>")
    if not notable:
        parts.append("<li><b>Look for signal elsewhere.</b> None of the relationships tested was reliable; additional variables (reviews, publication date) are needed.</li>")
    parts.append("</ul><h2>How far to trust this</h2><ul>"
                 "<li>Observational data from one site at one moment: associations are not causes.</li>"
                 "<li>Significance uses Holm-adjusted p-values across the four tests shown, and effect sizes must also be non-trivial to be called notable.</li>"
                 "<li>Extreme prices are excluded from the distribution and heat-map views, not from the statistics on price vs rating or category.</li></ul>")
    parts.append(f"<footer>Source: {esc(source)}. Charts: Matplotlib + Seaborn. Palette: colour-blind-safe (Okabe-Ito). Every headline is generated from the data.</footer>")
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Book catalogue data story</title><style>{STORY_CSS}</style></head><body>{"".join(parts)}</body></html>'


#main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=Path("../eda_project/eda_output/cleaned_books.csv"))
    ap.add_argument("--raw", type=Path, default=None, help="raw scrape (enables the data-quality chart)")
    ap.add_argument("--out", type=Path, default=Path("viz_output"))
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    setup_style()

    df = load(a.input)
    raw = None
    if a.raw:
        try:
            raw = pd.read_csv(a.raw, encoding="utf-8-sig")
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
            log.warning("Could not read raw file (%s); skipping the data-quality chart", exc)
    if len(df) < 20:
        log.warning("Only %d rows - charts will be sparse and intervals wide", len(df))

    synthetic = "synthetic" in a.input.name.lower() or (a.raw is not None and "synthetic" in a.raw.name.lower())
    footer = ("SYNTHETIC DEMO DATA · " if synthetic else "") + f"Source: books.toscrape.com scrape ({a.input.name}) · n = {len(df):,}"
    f = compute_facts(df, raw)
    I = insights(f)

    charts_dir = a.out / "charts"
    try:
        charts_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.error("Cannot create output folder: %s", exc)
        return 2

    made = 0
    for i, (key, draw, ok) in enumerate(CHARTS, 1):
        if key not in I or not ok(f):
            log.info("Skipping chart '%s' (not enough data)", key)
            continue
        try:
            fig = standalone(draw, df, f, I[key], footer)
            fig.savefig(charts_dir / f"{i:02d}_{key}.png", dpi=200)
            plt.close(fig); made += 1
        except Exception as exc:
            plt.close("all")
            log.warning("Chart '%s' failed: %s", key, exc)
    try:
        fig = dashboard(df, f, I, footer)
        fig.savefig(a.out / "dashboard.png", dpi=130); plt.close(fig)
        story = build_story(f, I, charts_dir, a.out / "dashboard.png", synthetic, a.input.name)
        (a.out / "data_story.html").write_text(story, encoding="utf-8")
        tableau_extract(df, f, a.out / "tableau_ready.csv")
    except Exception as exc:
        log.error("Could not build dashboard/story: %s", exc)
        return 1
    for k, v in I.items():
        log.info("%-13s %s", k, v["headline"])
    log.info("Done: %d charts + dashboard + story + Tableau extract -> %s", made, a.out.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
