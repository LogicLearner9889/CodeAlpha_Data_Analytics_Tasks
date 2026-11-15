# Task 2 - Exploratory Data Analysis (EDA)

Analyses the dataset scraped in Task 1 (`books.csv`) and produces a self-contained HTML report.

## Run
```bash
pip install -r requirements.txt
python eda.py --input ../web_scraper_project/output/books.csv --out eda_output
python make_sample_data.py            # optional: SYNTHETIC data for testing only
python -m unittest discover -s . -p "test_eda.py" -v
```
Open `eda_output/report.html` in any browser.

> `sample_data/books_SYNTHETIC.csv` and the report in `eda_output/` were generated from FAKE data with planted
> problems, to prove the pipeline works. They say nothing about the real website. Run on your own scrape.

## EDA workflow implemented
| Step | Where in the report |
|---|---|
| 1. Ask questions first | "Questions we want to answer" (Q1-Q8) |
| 2. Structure, types, completeness | Q1: dtype / missing / unique table + missingness chart |
| 3. Trends, patterns, anomalies | Q2-Q7: distributions, boxplots, Spearman heatmap, scatter, IQR outliers |
| 4. Hypotheses and assumptions | 7 tests with explicit H0, effect sizes, Holm correction; normality check justifies non-parametric tests |
| 5. Data issues | Q8: severity-ranked table with recommended action, plus `cleaned_books.csv` |

## Statistics used (and why)
- **Kruskal-Wallis / Mann-Whitney / Spearman**: rank-based; no normality assumption (prices are not normal and may have outliers).
- **Chi-square**: independence (rating vs stock) and goodness-of-fit (are ratings balanced?).
- **Effect sizes** (epsilon-squared, rho, rank-biserial, Cramer's V, Cohen's w): p-values alone mislead on large samples.
- **Holm correction**: controls false positives from running 7 tests at once.
- Tests are skipped with a reason when groups are too small; a failing test or chart never aborts the report.

## Outputs (`eda_output/`)
`report.html`, `figures/*.png`, `data_issues.csv`, `tests.csv`, `summary_stats.csv`, `cleaned_books.csv`

## Cleaning rules (cleaned_books.csv)
De-duplicate by URL; drop price <= 0 or missing; ratings outside 1-5 become missing; merge category spelling variants
into the most common form; add `desc_len`; **flag** (not delete) price outliers.

## Limits
Observational data from one site at one point in time: associations are not causal. Findings only generalise to this catalogue.
