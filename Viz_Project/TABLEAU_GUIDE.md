# Rebuilding the dashboard in Tableau (Public is free)

1. **Connect** -> Text file -> `viz_output/tableau_ready.csv`. Set `price_gbp` to Number (decimal), `rating` to Number (whole),
   `category`, `price_band`, `rating_label`, `in_stock` to String.
2. **Sheets** (one per chart):
   - *Price distribution*: Columns `price_gbp` -> right-click -> Create Bins (size 3) -> Rows `CNT(title)`. Colour bins inside IQR blue, rest grey (calculated field `[price_gbp] >= {FIXED : PERCENTILE([price_gbp],0.25)} AND [price_gbp] <= {FIXED : PERCENTILE([price_gbp],0.75)}`).
   - *Rating balance*: Columns `rating_label`, Rows `CNT(title)`, add a Reference Line = average (shows the "even split").
   - *Price by rating*: Columns `rating_label`, Rows `MEDIAN(price_gbp)`; Analytics pane -> Box Plot or drag Distribution Band.
   - *Category*: Rows `category` sorted by median price, Columns `MEDIAN(price_gbp)`, Label `CNT(title)`; filter `CNT(title) >= 15`.
   - *Heat map*: Rows `rating_label`, Columns `price_band`, Marks = Square, Colour `CNT(title)` (blue sequential), Label `CNT(title)`.
3. **Dashboard**: 1200x800 -> top row KPI text tiles (`COUNT`, `MEDIAN(price)`, `AVG(rating)`), then the sheets in a 3x2 grid.
   Add a `category` and `in_stock` filter applied to all sheets.
4. **Style**: accent `#0072B2`, grey `#9AA0A6`, flag `#D55E00`, font Tableau Book; write each sheet title as the finding, not the field names.
5. **Publish** to Tableau Public and link it in your portfolio.
