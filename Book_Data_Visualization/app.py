from pathlib import Path
import logging
import sys
import traceback

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------- CONFIGURATION -------------------------
APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "cleaned_books.csv"
OUTPUT_DIR = APP_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=APP_DIR / "dashboard_errors.log",
    level=logging.ERROR,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

st.set_page_config(
    page_title="Book Data Analytics Dashboard",
    #page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Book Data Visualization Dashboard")
st.caption("TASK 3 | Data analysis, visualization and data storytelling")

# ------------------------- ERROR HANDLING -------------------------
class DatasetError(Exception):
    """Raised when the dataset is missing or unusable."""


@st.cache_data(show_spinner="Loading and validating dataset...")
def load_data(file_path: str, uploaded_file=None) -> pd.DataFrame:
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            path = Path(file_path)
            if not path.exists():
                raise DatasetError(
                    f"Dataset not found: {path}. Put cleaned_books.csv inside the data folder."
                )
            if path.stat().st_size == 0:
                raise DatasetError("The CSV file is empty.")
            df = pd.read_csv(path, encoding="utf-8")

        if df.empty:
            raise DatasetError("The dataset contains no rows.")

        if len(df.columns) == 0:
            raise DatasetError("No columns were found in the CSV file.")

        df.columns = [str(c).strip().lower() for c in df.columns]

        required = ["title", "category", "price_gbp", "rating", "in_stock", "stock_count"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise DatasetError("Required columns are missing: " + ", ".join(missing))

        # Standardize text fields
        for col in ["title", "category", "description", "url", "upc"]:
            if col in df.columns:
                df[col] = df[col].astype("string").str.strip()

        # Convert numeric fields safely
        numeric_cols = ["price_gbp", "rating", "stock_count", "num_reviews", "desc_len"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Convert stock flags safely
        if df["in_stock"].dtype != bool:
            truth_map = {
                "true": True, "1": True, "yes": True, "in stock": True,
                "false": False, "0": False, "no": False, "out of stock": False
            }
            df["in_stock"] = (
                df["in_stock"].astype("string").str.strip().str.lower()
                .map(truth_map)
            )

        # Parse timestamp if available
        if "scraped_at" in df.columns:
            df["scraped_at"] = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)

        # Clean impossible / invalid values without silently changing valid values
        df.loc[df["price_gbp"] < 0, "price_gbp"] = np.nan
        df.loc[~df["rating"].between(1, 5), "rating"] = np.nan
        df.loc[df["stock_count"] < 0, "stock_count"] = np.nan
        if "num_reviews" in df.columns:
            df.loc[df["num_reviews"] < 0, "num_reviews"] = np.nan

        # Fill text fields only where appropriate
        df["title"] = df["title"].fillna("Unknown title")
        df["category"] = df["category"].fillna("Uncategorized")
        if "description" in df.columns:
            df["description"] = df["description"].fillna("")

        # Remove exact duplicate rows
        df = df.drop_duplicates().reset_index(drop=True)

        # Derived fields
        if "price_outlier" not in df.columns:
            df["price_outlier"] = False
        else:
            df["price_outlier"] = (
                df["price_outlier"].astype("string").str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
                .fillna(False)
            )

        if "desc_len" not in df.columns and "description" in df.columns:
            df["desc_len"] = df["description"].str.len()
        elif "desc_len" in df.columns and "description" in df.columns:
            df["desc_len"] = df["desc_len"].fillna(df["description"].str.len())

        return df

    except DatasetError:
        raise
    except pd.errors.EmptyDataError as exc:
        raise DatasetError("The CSV file has no readable data.") from exc
    except pd.errors.ParserError as exc:
        raise DatasetError(f"CSV parsing failed: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise DatasetError("Text encoding problem. Save the CSV as UTF-8 and try again.") from exc
    except Exception as exc:
        logging.exception("Unexpected dataset loading error")
        raise DatasetError(f"Unexpected loading error: {exc}") from exc


def safe_mean(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else 0.0


def save_figure(fig, filename):
    try:
        fig.savefig(OUTPUT_DIR / filename, dpi=180, bbox_inches="tight")
    except Exception:
        logging.exception("Could not save figure: %s", filename)


# ------------------------- DATA INPUT -------------------------
with st.sidebar:
    st.header("Dashboard Controls")
    uploaded = st.file_uploader("Upload another CSV (optional)", type=["csv"])
    st.caption("Default dataset: data/cleaned_books.csv")

try:
    df = load_data(str(DATA_PATH), uploaded)
except DatasetError as error:
    st.error(str(error))
    st.info("Check the file path, CSV format and required column names.")
    st.stop()
except Exception as error:
    logging.exception("Fatal application error")
    st.error("The dashboard could not start because of an unexpected error.")
    st.code(str(error))
    st.stop()

# ------------------------- SIDEBAR FILTERS -------------------------
with st.sidebar:
    st.subheader("Filters")
    categories = sorted(df["category"].dropna().astype(str).unique().tolist())
    selected_categories = st.multiselect(
        "Book category", categories, default=categories
    )

    min_price = float(df["price_gbp"].min()) if df["price_gbp"].notna().any() else 0.0
    max_price = float(df["price_gbp"].max()) if df["price_gbp"].notna().any() else 100.0
    if min_price == max_price:
        price_range = (min_price, max_price)
        st.caption(f"Only one valid price value: £{min_price:.2f}")
    else:
        price_range = st.slider(
            "Price range (₹)", min_value=min_price, max_value=max_price,
            value=(min_price, max_price)
        )

    stock_filter = st.selectbox(
        "Availability", ["All", "In stock", "Out of stock"]
    )
    search_text = st.text_input("Search book title", placeholder="Enter a title...")

filtered = df.copy()
if selected_categories:
    filtered = filtered[filtered["category"].astype(str).isin(selected_categories)]
else:
    filtered = filtered.iloc[0:0]

filtered = filtered[
    filtered["price_gbp"].isna()
    | filtered["price_gbp"].between(price_range[0], price_range[1])
]
if stock_filter == "In stock":
    filtered = filtered[filtered["in_stock"] == True]
elif stock_filter == "Out of stock":
    filtered = filtered[filtered["in_stock"] == False]
if search_text:
    filtered = filtered[
        filtered["title"].astype(str).str.contains(search_text, case=False, na=False)
    ]

st.caption(f"Showing {len(filtered):,} of {len(df):,} books after filters.")

# ------------------------- KPI CARDS -------------------------
total_books = len(filtered)
avg_price = safe_mean(filtered["price_gbp"])
avg_rating = safe_mean(filtered["rating"])
in_stock_count = int((filtered["in_stock"] == True).sum())
category_count = filtered["category"].nunique()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Books", f"{total_books:,}")
k2.metric("Average Price", f"₹{avg_price:,.2f}")
k3.metric("Average Rating", f"{avg_rating:.2f}/5")
k4.metric("In Stock", f"{in_stock_count:,}")
k5.metric("Categories", f"{category_count:,}")

if filtered.empty:
    st.warning("No books match the selected filters. Change or clear the filters.")
    st.stop()

# ------------------------- TABS -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Categories", "Price Analysis",
    "Ratings & Stock", "Data Explorer"
])

sns.set_theme(style="whitegrid")

with tab1:
    st.subheader("Dataset Overview")
    left, right = st.columns(2)

    with left:
        counts = filtered["category"].value_counts().head(12).sort_values()
        fig, ax = plt.subplots(figsize=(9, 5))
        counts.plot(kind="barh", ax=ax)
        ax.set_title("Top Categories by Number of Books")
        ax.set_xlabel("Number of books")
        ax.set_ylabel("Category")
        st.pyplot(fig)
        save_figure(fig, "category_counts.png")
        plt.close(fig)

    with right:
        stock_counts = filtered["in_stock"].map(
            {True: "In Stock", False: "Out of Stock"}
        ).fillna("Unknown").value_counts()
        fig, ax = plt.subplots(figsize=(7, 5))
        if not stock_counts.empty:
            ax.pie(stock_counts.values, labels=stock_counts.index,
                   autopct="%1.1f%%", startangle=90)
            ax.set_title("Book Availability")
        st.pyplot(fig)
        save_figure(fig, "availability.png")
        plt.close(fig)

    st.markdown("### Quick observations")
    most_common = filtered["category"].mode()
    if not most_common.empty:
        st.write(f"- Most represented category: **{most_common.iloc[0]}**.")
    if filtered["price_gbp"].notna().any():
        cheapest = filtered.loc[filtered["price_gbp"].idxmin()]
        costliest = filtered.loc[filtered["price_gbp"].idxmax()]
        st.write(f"- Lowest listed price: **{cheapest['title']} (₹{cheapest['price_gbp']:.2f})**.")
        st.write(f"- Highest listed price: **{costliest['title']} (₹{costliest['price_gbp']:.2f})**.")
    st.write(f"- Books currently marked in stock: **{in_stock_count}**.")

with tab2:
    st.subheader("Category Analysis")
    category_summary = (
        filtered.groupby("category", dropna=False)
        .agg(
            book_count=("title", "count"),
            average_price=("price_gbp", "mean"),
            average_rating=("rating", "mean"),
            total_stock=("stock_count", "sum")
        )
        .reset_index()
        .sort_values("book_count", ascending=False)
    )
    st.dataframe(category_summary, use_container_width=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    plot_data = category_summary.sort_values("book_count", ascending=False).head(15)
    sns.barplot(data=plot_data, x="book_count", y="category", ax=ax)
    ax.set_title("Book Count by Category (Top 15)")
    ax.set_xlabel("Number of books")
    ax.set_ylabel("Category")
    st.pyplot(fig)
    save_figure(fig, "books_by_category.png")
    plt.close(fig)

    rating_data = category_summary.dropna(subset=["average_rating"])
    if not rating_data.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.barplot(data=rating_data.sort_values("average_rating", ascending=False).head(15),
                    x="average_rating", y="category", ax=ax)
        ax.set_xlim(0, 5)
        ax.set_title("Average Rating by Category")
        st.pyplot(fig)
        save_figure(fig, "average_rating_by_category.png")
        plt.close(fig)

with tab3:
    st.subheader("Price Analysis")
    price_data = filtered["price_gbp"].dropna()
    if price_data.empty:
        st.warning("No valid price values are available for the selected filters.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(price_data, bins=20, kde=True, ax=ax)
            ax.set_title("Book Price Distribution")
            ax.set_xlabel("Price (£)")
            st.pyplot(fig)
            save_figure(fig, "price_distribution.png")
            plt.close(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(x=price_data, ax=ax)
            ax.set_title("Book Price Outlier Overview")
            ax.set_xlabel("Price (£)")
            st.pyplot(fig)
            save_figure(fig, "price_boxplot.png")
            plt.close(fig)

        st.write("**Price statistics**")
        st.dataframe(price_data.describe().to_frame("price_gbp"), use_container_width=True)

        outliers = filtered[filtered["price_outlier"] == True]
        st.metric("Rows flagged as price outliers", len(outliers))
        if not outliers.empty:
            st.dataframe(outliers[["title", "category", "price_gbp", "price_outlier"]],
                         use_container_width=True)

with tab4:
    st.subheader("Ratings and Inventory")
    c1, c2 = st.columns(2)
    rating_data = filtered["rating"].dropna()
    with c1:
        if rating_data.empty:
            st.info("No valid ratings to display.")
        else:
            fig, ax = plt.subplots(figsize=(7, 5))
            rating_order = sorted(rating_data.unique())
            sns.countplot(x=rating_data, order=rating_order, ax=ax)
            ax.set_title("Rating Distribution")
            ax.set_xlabel("Rating (1–5)")
            ax.set_ylabel("Number of books")
            st.pyplot(fig)
            save_figure(fig, "rating_distribution.png")
            plt.close(fig)

    with c2:
        stock_data = filtered.dropna(subset=["stock_count", "price_gbp"])
        if stock_data.empty:
            st.info("No stock and price pairs available.")
        else:
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.scatterplot(data=stock_data, x="stock_count", y="price_gbp",
                            hue="in_stock", ax=ax)
            ax.set_title("Stock Quantity vs Book Price")
            ax.set_xlabel("Stock quantity")
            ax.set_ylabel("Price (₹)")
            st.pyplot(fig)
            save_figure(fig, "stock_vs_price.png")
            plt.close(fig)

    if "num_reviews" in filtered.columns:
        st.markdown("### Reviews")
        st.write(f"Total reviews recorded: **{int(filtered['num_reviews'].fillna(0).sum()):,}**")
        review_data = filtered.dropna(subset=["num_reviews", "rating"])
        if not review_data.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=review_data, x="num_reviews", y="rating", ax=ax)
            ax.set_title("Number of Reviews vs Rating")
            st.pyplot(fig)
            save_figure(fig, "reviews_vs_rating.png")
            plt.close(fig)

with tab5:
    st.subheader("Searchable Book Records")
    display_cols = [
        c for c in [
            "title", "category", "price_gbp", "rating", "in_stock",
            "stock_count", "num_reviews", "price_outlier"
        ] if c in filtered.columns
    ]
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data as CSV",
        data=csv_bytes,
        file_name="filtered_books.csv",
        mime="text/csv"
    )

# ------------------------- DATA QUALITY REPORT -------------------------
with st.expander("Data quality report"):
    quality = pd.DataFrame({
        "Column": df.columns,
        "Missing values": [int(df[c].isna().sum()) for c in df.columns],
        "Data type": [str(df[c].dtype) for c in df.columns],
        "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns]
    })
    st.dataframe(quality, use_container_width=True, hide_index=True)
    st.write(f"Rows after exact duplicate removal: **{len(df):,}**")
    st.write(f"Rows currently displayed after filtering: **{len(filtered):,}**")

# ------------------------- EXPORT CLEANED DATA -------------------------
try:
    export_df = df.copy()
    export_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download validated dataset",
        data=export_bytes,
        file_name="validated_cleaned_books.csv",
        mime="text/csv"
    )
except Exception:
    logging.exception("Dataset export failed")
    st.error("Could not prepare the cleaned dataset download.")

st.divider()
st.caption("Book Data Visualization | Built with Python, Pandas, Matplotlib, Seaborn and Streamlit")
