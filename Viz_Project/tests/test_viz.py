import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import viz  # noqa: E402

CATS = ["Fiction", "Nonfiction", "Travel", "Mystery", "Romance", "Poetry"]


def make_df(n=900, seed=0, with_category=True):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "title": [f"Book {i}" for i in range(n)], "url": [f"http://x/{i}" for i in range(n)],
        "price_gbp": np.round(rng.uniform(10, 60, n), 2), "rating": rng.integers(1, 6, n),
        "in_stock": rng.random(n) < .95, "stock_count": rng.integers(1, 20, n).astype(float),
        "desc_len": rng.integers(50, 600, n), "detail_ok": True, "price_outlier": False,
    })
    if with_category:
        df["category"] = rng.choice(CATS, n)
    return df


def facts_for(df):
    f = viz.compute_facts(viz.load(_write(df)))
    return f, viz.insights(f)


def _write(df):
    p = Path(tempfile.mkdtemp()) / "books.csv"
    df.to_csv(p, index=False)
    return p


class TruthfulHeadlineTests(unittest.TestCase):
    def test_null_data_gives_neutral_story(self):
        f, I = facts_for(make_df(seed=1))
        self.assertFalse(any(t["notable"] for t in f["tests"].values()), f["tests"])
        self.assertIn("do not cost more", I["price_rating"]["headline"])
        self.assertIn("evenly", I["rating"]["headline"])
        self.assertIn("within noise", I["category"]["headline"])

    def test_positive_price_rating_link(self):
        df = make_df(seed=1); df["price_gbp"] += df["rating"] * 4
        f, I = facts_for(df)
        self.assertTrue(f["tests"]["price_rating"]["notable"])
        self.assertIn("rises", I["price_rating"]["headline"])

    def test_negative_price_rating_link(self):
        df = make_df(seed=1); df["price_gbp"] -= df["rating"] * 4; df["price_gbp"] += 25
        _, I = facts_for(df)
        self.assertIn("falls", I["price_rating"]["headline"])

    def test_category_effect_named(self):
        df = make_df(seed=1); df.loc[df["category"] == "Travel", "price_gbp"] += 25
        f, I = facts_for(df)
        self.assertTrue(f["tests"]["category"]["notable"])
        self.assertIn("Travel is the priciest", I["category"]["headline"])

    def test_rating_imbalance_named(self):
        df = make_df(seed=1); df.loc[df.index[:500], "rating"] = 5
        _, I = facts_for(df)
        self.assertIn("5-star is the most common", I["rating"]["headline"])

    def test_negligible_effect_not_called_notable(self):
        """Huge n makes tiny effects 'significant'; they must not be headlined as findings."""
        df = make_df(n=60000, seed=2); df["price_gbp"] += df["rating"] * 0.05
        f, I = facts_for(df)
        self.assertFalse(f["tests"]["price_rating"]["notable"])
        self.assertIn("do not cost more", I["price_rating"]["headline"])

    def test_price_takeaway_flat_distribution(self):
        _, I = facts_for(make_df(seed=1))
        self.assertIn("no single 'typical' price", I["price"]["takeaway"])

    def test_holm(self):
        adj = viz.holm([0.01, 0.04, 0.03])
        self.assertAlmostEqual(adj[0], 0.03); self.assertTrue(np.all(adj <= 1))

    def test_bootstrap_ci_contains_median(self):
        x = np.random.default_rng(0).normal(50, 5, 300)
        lo, hi = viz.boot_median_ci(x)
        self.assertTrue(lo < np.median(x) < hi)
        self.assertTrue(all(np.isnan(viz.boot_median_ci([1, 2]))))


class PipelineTests(unittest.TestCase):
    def run_main(self, df=None, name="books.csv", raw=None, out=None):
        d = Path(tempfile.mkdtemp())
        src = d / name
        df.to_csv(src, index=False)
        out = out or d / "out"
        args = ["--input", str(src), "--out", str(out)]
        if raw is not None:
            rp = d / "raw.csv"; raw.to_csv(rp, index=False); args += ["--raw", str(rp)]
        return viz.main(args), out

    def test_full_outputs_and_valid_pngs(self):
        raw = make_df(seed=3); raw = pd.concat([raw, raw.head(10)])
        code, out = self.run_main(make_df(seed=3), raw=raw)
        self.assertEqual(code, 0)
        pngs = sorted((out / "charts").glob("*.png"))
        self.assertEqual(len(pngs), 6)
        for p in pngs + [out / "dashboard.png"]:
            data = p.read_bytes()
            self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n"); self.assertGreater(len(data), 10_000)
        story = (out / "data_story.html").read_text(encoding="utf-8")
        self.assertIn("alt=", story); self.assertIn("So what", story)
        self.assertNotIn("Synthetic demo data", story)

    def test_synthetic_filename_is_labelled(self):
        _, out = self.run_main(make_df(seed=3), name="books_SYNTHETIC.csv")
        self.assertIn("Synthetic demo data", (out / "data_story.html").read_text(encoding="utf-8"))

    def test_without_raw_skips_quality_chart(self):
        code, out = self.run_main(make_df(seed=3))
        self.assertEqual(code, 0)
        self.assertEqual(len(list((out / "charts").glob("*.png"))), 5)
        self.assertFalse(any("quality" in p.name for p in (out / "charts").glob("*.png")))

    def test_no_category_column(self):
        code, out = self.run_main(make_df(seed=3, with_category=False))
        self.assertEqual(code, 0)
        self.assertTrue((out / "dashboard.png").exists())
        self.assertFalse(any("category" in p.name for p in (out / "charts").glob("*.png")))

    def test_tiny_dataset(self):
        code, out = self.run_main(make_df(n=12, seed=3))
        self.assertEqual(code, 0)
        self.assertTrue((out / "dashboard.png").exists())

    def test_constant_columns(self):
        df = make_df(seed=3); df["rating"] = 3; df["price_gbp"] = 20.0
        code, _ = self.run_main(df)
        self.assertEqual(code, 0)

    def test_extreme_outlier_does_not_flatten_charts(self):
        df = make_df(seed=3); df.loc[0, "price_gbp"] = 99999
        f, _ = facts_for(df)
        self.assertEqual(f["n_outliers"], 1)
        self.assertLess(f["pmax"], 100)

    def test_tableau_extract(self):
        df = make_df(seed=3)
        _, out = self.run_main(df)
        t = pd.read_csv(out / "tableau_ready.csv")
        self.assertEqual(len(t), len(df))
        for c in ("price_band", "rating_label", "category", "in_stock"):
            self.assertIn(c, t.columns)
        self.assertTrue(set(t["in_stock"]) <= {"In stock", "Out of stock", "Unknown"})

    def test_unwritable_output(self):
        d = Path(tempfile.mkdtemp()); blocker = d / "file"; blocker.write_text("x")
        code, _ = self.run_main(make_df(seed=3), out=blocker / "sub")
        self.assertEqual(code, 2)

    def test_bad_raw_file_is_non_fatal(self):
        d = Path(tempfile.mkdtemp()); src = d / "books.csv"; make_df(seed=3).to_csv(src, index=False)
        (d / "raw.csv").write_text("")
        self.assertEqual(viz.main(["--input", str(src), "--raw", str(d / "raw.csv"), "--out", str(d / "o")]), 0)


class InputErrorTests(unittest.TestCase):
    def test_missing_file(self):
        with self.assertRaises(SystemExit) as cm:
            viz.load(Path("/no/such.csv"))
        self.assertIn("not found", str(cm.exception))

    def test_empty_file(self):
        p = Path(tempfile.mkdtemp()) / "e.csv"; p.write_text("")
        with self.assertRaises(SystemExit):
            viz.load(p)

    def test_missing_columns(self):
        p = Path(tempfile.mkdtemp()) / "m.csv"; p.write_text("a,b\n1,2\n")
        with self.assertRaises(SystemExit) as cm:
            viz.load(p)
        self.assertIn("missing required columns", str(cm.exception))

    def test_no_valid_prices(self):
        p = Path(tempfile.mkdtemp()) / "n.csv"; p.write_text("price_gbp,rating\nabc,3\n-5,2\n")
        with self.assertRaises(SystemExit):
            viz.load(p)


if __name__ == "__main__":
    unittest.main()
