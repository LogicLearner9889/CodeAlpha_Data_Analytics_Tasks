import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import eda
from make_sample_data import make


def prepared(df):
    return eda.clean(df)


class StatisticalPowerTests(unittest.TestCase):
    """The tests must find real effects and stay quiet on pure noise."""

    def test_null_data_no_false_positives(self):
        res = eda.run_tests(prepared(make(n=1500, seed=3, inject_issues=False)))
        self.assertEqual(res["verdict"].str.startswith("Reject").sum(), 0)

    def test_detects_planted_rating_price_effect(self):
        df = make(n=1500, seed=3, inject_issues=False)
        df["price_gbp"] = df["price_gbp"] + df["rating"] * 4          # higher rating -> pricier
        res = eda.run_tests(prepared(df)).set_index("name")
        for name in ("Price differs by rating (Kruskal-Wallis)", "Price vs rating association (Spearman)"):
            self.assertTrue(res.loc[name, "verdict"].startswith("Reject"), name)
            self.assertNotEqual(res.loc[name, "effect_size"], "negligible")

    def test_detects_planted_category_effect(self):
        df = make(n=1500, seed=3, inject_issues=False)
        df.loc[df["category"] == "Travel", "price_gbp"] += 25
        res = eda.run_tests(prepared(df)).set_index("name")
        self.assertTrue(res.loc["Price differs by category (Kruskal-Wallis)", "verdict"].startswith("Reject"))

    def test_detects_unbalanced_ratings(self):
        df = make(n=1500, seed=3, inject_issues=False)
        df.loc[df.index[:600], "rating"] = 5
        res = eda.run_tests(prepared(df)).set_index("name")
        self.assertTrue(res.loc["Ratings are balanced (chi-square goodness-of-fit)", "verdict"].startswith("Reject"))

    def test_holm_is_monotone_and_bounded(self):
        p = np.array([0.01, 0.04, 0.03, 0.5])
        adj = eda.holm(p)
        self.assertTrue(np.all(adj >= p) and np.all(adj <= 1))
        self.assertAlmostEqual(adj[0], 0.04)       # 4 * 0.01

    def test_effect_labels(self):
        self.assertEqual(eda.label_effect("corr", 0.05), "negligible")
        self.assertEqual(eda.label_effect("corr", -0.35), "medium")
        self.assertEqual(eda.label_effect("eps2", 0.2), "large")


class IssueDetectionTests(unittest.TestCase):
    def setUp(self):
        self.raw = make(n=500, seed=1)
        self.issues = eda.detect_issues(self.raw, [])

    def has(self, text):
        return self.issues["issue"].str.contains(text, regex=False).any()

    def test_all_planted_problems_found(self):
        for text in ("Duplicate rows", "Price <= 0", "Rating outside", "Price outliers", "Detail page failed",
                     "out of stock but stock_count", "spelling/whitespace", "Constant column 'num_reviews'"):
            self.assertTrue(self.has(text), text)

    def test_exact_counts(self):
        row = lambda t: int(self.issues.loc[self.issues["issue"].str.contains(t, regex=False), "rows_affected"].iloc[0])
        self.assertEqual(row("Duplicate rows"), 12)
        self.assertEqual(row("Price <= 0"), 1)
        self.assertEqual(row("Rating outside"), 1)
        self.assertEqual(row("out of stock but stock_count"), 3)

    def test_clean_data_has_no_high_issues(self):
        clean_raw = make(n=300, seed=5, inject_issues=False)
        issues = eda.detect_issues(clean_raw, [])
        self.assertEqual((issues["severity"] == "High").sum(), 0)

    def test_sorted_by_severity(self):
        order = {"High": 0, "Medium": 1, "Low": 2, "Info": 3}
        sev = self.issues["severity"].map(order).tolist()
        self.assertEqual(sev, sorted(sev))


class CleaningTests(unittest.TestCase):
    def test_cleaning_rules(self):
        c = eda.clean(make(n=500, seed=1))
        self.assertFalse(c["url"].duplicated().any())
        self.assertTrue((c["price_gbp"] > 0).all())
        self.assertTrue(c["rating"].dropna().between(1, 5).all())
        self.assertTrue(c["price_outlier"].any())                   # flagged, not removed
        self.assertIn(999.99, c["price_gbp"].values)
        self.assertEqual(c.loc[c["category"].fillna("").str.lower().str.strip() == "poetry", "category"].nunique(), 1)


class RobustnessTests(unittest.TestCase):
    def run_main(self, csv_text=None, df=None):
        d = tempfile.mkdtemp()
        src = Path(d) / "in.csv"
        if df is not None:
            df.to_csv(src, index=False)
        elif csv_text is not None:
            src.write_text(csv_text)
        return eda.main(["--input", str(src), "--out", str(Path(d) / "out")]), Path(d) / "out"

    def test_missing_file_exits_cleanly(self):
        with self.assertRaises(SystemExit) as cm:
            eda.main(["--input", "/nonexistent.csv", "--out", tempfile.mkdtemp()])
        self.assertIn("not found", str(cm.exception))

    def test_empty_file(self):
        with self.assertRaises(SystemExit):
            self.run_main(csv_text="")

    def test_missing_required_columns(self):
        with self.assertRaises(SystemExit) as cm:
            self.run_main(csv_text="a,b\n1,2\n")
        self.assertIn("missing required columns", str(cm.exception))

    def test_tiny_dataset_skips_tests_not_crashes(self):
        code, out = self.run_main(df=make(n=8, seed=2, inject_issues=False))
        self.assertEqual(code, 0)
        self.assertTrue((out / "report.html").exists())
        t = pd.read_csv(out / "tests.csv")
        self.assertTrue(t["note"].fillna("").str.contains("SKIPPED").any())

    def test_dirty_types_are_coerced_and_reported(self):
        df = make(n=200, seed=2, inject_issues=False).astype({"price_gbp": object, "rating": object})
        df.loc[0, "price_gbp"] = "N/A"
        df.loc[1, "rating"] = "five"
        code, out = self.run_main(df=df)
        self.assertEqual(code, 0)
        issues = pd.read_csv(out / "data_issues.csv")
        self.assertTrue(issues["issue"].str.contains("Non-numeric").any())

    def test_constant_price_does_not_crash(self):
        df = make(n=100, seed=2, inject_issues=False)
        df["price_gbp"] = 20.0
        code, _ = self.run_main(df=df)
        self.assertEqual(code, 0)

    def test_full_run_outputs(self):
        code, out = self.run_main(df=make(n=400, seed=2))
        self.assertEqual(code, 0)
        for f in ("report.html", "data_issues.csv", "tests.csv", "summary_stats.csv", "cleaned_books.csv"):
            self.assertTrue((out / f).exists(), f)
        self.assertGreaterEqual(len(list((out / "figures").glob("*.png"))), 6)

    def test_unwritable_output_returns_error_code(self):
        d = tempfile.mkdtemp()
        blocker = Path(d) / "file"
        blocker.write_text("x")                        # a file where a directory is needed
        src = Path(d) / "in.csv"
        make(n=100, seed=2).to_csv(src, index=False)
        self.assertEqual(eda.main(["--input", str(src), "--out", str(blocker / "sub")]), 2)


if __name__ == "__main__":
    unittest.main()
