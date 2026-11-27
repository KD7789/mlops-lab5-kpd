# Data

`online_shoppers_intention.csv` is the Online Shoppers Purchasing Intention Dataset from the UCI Machine Learning Repository.

- Source: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- Citation: Sakar, C. & Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q
- License: CC BY 4.0, which allows sharing the file here as long as the source is credited.
- Downloaded with `curl -Lf -o data/online_shoppers_intention.csv 'https://archive.ics.uci.edu/static/public/468/data.csv'`

The file has 12,330 sessions (visits to an online store over one year) and 18 columns. It has no missing values and 125 exact duplicate rows. Months run from Feb to Dec, with no Jan or Apr.

How this lab uses it:

- Sessions from Feb to Oct (7,605 rows) are the training data. Sessions from Nov and Dec (4,725 rows) play the role of new sessions, which the scoring DAG assigns to segments.
- Clustering uses 9 browsing-behaviour columns: `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates` and `PageValues`.
- `Revenue` (whether the session ended in a purchase) is never used for training. It is only used to report each segment's purchase rate.
