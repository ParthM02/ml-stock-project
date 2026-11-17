# Data files

Download the Kaggle **New York Stock Exchange** dataset and place these files in this directory:

- `prices-split-adjusted.csv`
- `securities.csv`

The notebook intentionally keeps raw data out of version control. The expected Kaggle page is:
https://www.kaggle.com/datasets/dgawlik/nyse

The current local copy is Kaggle dataset version 3, downloaded on 2026-11-16.

It can be downloaded with:

```python
import kagglehub

path = kagglehub.dataset_download("dgawlik/nyse")
print("Path to dataset files:", path)
```

