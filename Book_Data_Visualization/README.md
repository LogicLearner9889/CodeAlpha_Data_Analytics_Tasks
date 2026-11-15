# Book Data Visualization Dashboard

An interactive dashboard built using the supplied `cleaned_books.csv` dataset.

## Features
- KPI cards: total books, average price, average rating, in-stock count, categories
- Sidebar filters: category, price range, availability, title search
- Category-wise book counts and ratings
- Price distribution, boxplot and flagged outlier inspection
- Rating distribution and stock analysis
- Searchable data table and CSV downloads
- Data quality report
- Defensive loading, validation, logging and user-facing error messages

## 1. Requirements
- Python 3.10 or newer recommended
- Internet connection for the first-time package installation
- VS Code or another terminal

## 2. Open the project
Extract the ZIP file. Open the `TASK_3_Book_Data_Visualization` folder in VS Code.

## 3. Create a virtual environment

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

If PowerShell blocks activation, use Command Prompt, or run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## 4. Install libraries
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Run the dashboard
From the project root folder:
```bash
python -m streamlit run app.py
```
Streamlit normally opens the dashboard in your browser. If it does not, open the Local URL printed in the terminal (usually `http://localhost:8501`).

## 6. Dataset location
The default dataset must be here:
```text
data/cleaned_books.csv
```
You can also upload another compatible CSV from the dashboard sidebar. Required columns:
`title`, `category`, `price_gbp`, `rating`, `in_stock`, `stock_count`.

## 7. Output
- Charts are displayed in the dashboard.
- PNG chart files are saved under `outputs/`.
- The dashboard provides downloads for filtered data and validated data.
- Unexpected errors are logged to `dashboard_errors.log`.

## 8. Common errors and fixes

### `python is not recognized`
Install Python and enable **Add Python to PATH**, then reopen the terminal.

### `No module named streamlit`
Activate the virtual environment and run:
```bash
python -m pip install -r requirements.txt
```

### `streamlit is not recognized`
Use:
```bash
python -m streamlit run app.py
```

### Dataset not found
Confirm the file path is exactly:
`data/cleaned_books.csv`

### Port 8501 is already in use
Run:
```bash
python -m streamlit run app.py --server.port 8502
```

### CSV parsing error
Open the CSV in Excel and save as **CSV UTF-8**, then upload it again.

### Dashboard shows no results
Clear the category, price, availability and title filters.

## 9. Suggested analysis questions
1. Which category contains the most books?
2. Which category has the highest average book price?
3. What is the distribution of book ratings?
4. How many books are currently in stock?
5. Which books are flagged as price outliers?
6. Is there an apparent relationship between stock quantity and price?

## 10. Important interpretation note
This file contains 200 book records and is a snapshot, not a time series of sales. Do not interpret it as sales performance or sales growth. `num_reviews` may be zero for all or many rows; if so, review-based conclusions are limited. The existing `price_outlier` field is used as supplied; it is not recalculated by this dashboard.
