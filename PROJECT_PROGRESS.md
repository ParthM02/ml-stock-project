# S&P 500 Stock-Direction Project: Work Completed

## Project objective

The project predicts whether each stock's split-adjusted closing price will rise on the next market session. One observation is a `(symbol, date)` pair, and one shared classifier is trained across all tickers.

The main research question is whether sector-relative features improve out-of-time predictions beyond a stock's own price and volume history. A secondary comparison tests logistic regression against gradient-boosted trees.

All analysis is contained in `sp500_stock_direction.ipynb` and runs from the two raw Kaggle CSV files.

## Environment and data

An isolated Python environment was created in `.venv`, with dependencies recorded in `requirements.txt`.

The data comes from version 3 of the Kaggle New York Stock Exchange dataset (`dgawlik/nyse`), downloaded on November 16, 2026. The notebook uses:

- `data/prices-split-adjusted.csv`
- `data/securities.csv`

The observed price data contains:

- 851,264 raw stock-day rows
- 501 tickers
- Dates from January 4, 2010 through December 30, 2016
- 1,762 distinct market sessions

The raw data contained no exact duplicates, conflicting `(symbol, date)` keys, invalid closes, negative volumes, or inconsistent OHLC ranges. All 501 tickers matched the supplied sector metadata.

The ticker universe is a later snapshot rather than a historically exact S&P 500 membership list. Only 467 tickers have prices on the first date, and other tickers enter the dataset later. Aligning all tickers to the complete session calendar exposes 31,498 missing stock-session observations. These observations remain missing so a gap cannot masquerade as a one-session return.

## Data and feature engineering

The notebook builds a complete ticker-by-session panel before shifting or rolling. Labels are valid only when both the current and next consecutive market sessions contain closing prices.

The retained target fields are:

- `target_date`
- `next_return = close[t+1] / close[t] - 1`
- `label = 1` when `next_return > 0`, otherwise `0`

`target_date` and `next_return` are retained for validation and analysis but are excluded from every model feature matrix.

### Stock-history features

- 1-session return
- 5-session return
- 20-session return
- Standard deviation of the most recent 20 one-session returns
- Distance from the trailing 20-session moving average
- Daily price range `(high - low) / close`
- Current volume relative to the mean of the previous 20 sessions

Every rolling calculation requires complete history for its window.

### Sector-relative features

- Stock 1-session return minus the mean return of its other sector peers
- Stock 5-session return minus the mean return of its other sector peers

The sector means exclude the current stock and require at least five valid peers. Sector aggregates are calculated before filtering on future-label availability.

Feature coverage is:

- 840,743 labeled rows with complete stock-only features
- 832,038 labeled rows with complete stock and sector features
- 8,705 rows lost by requiring the sector-relative features

All model feature-set comparisons use the same 832,038 complete rows.

## Engineering and leakage tests

The notebook includes both full-data assertions and deterministic tests on a small synthetic stock panel.

The synthetic tests verify:

1. Exact five-session return calculation
2. Prior-20-session relative-volume denominator
3. Twenty-return volatility
4. Leave-one-stock-out sector-relative return
5. Missing-session label invalidation
6. Minimum five-peer requirement

Additional checks enforce:

- Unique `(symbol, date)` keys
- Consecutive prediction and target sessions
- No infinite model inputs
- No identifiers, dates, targets, or raw ticker values in the model matrix
- Correct manual label and return calculations for Apple
- Missing labels immediately before a synthetic price gap
- Non-overlapping temporal splits

A future-mutation leakage test changes all prices and volumes strictly after June 30, 2014 and rebuilds the features. All features on the sample date remain unchanged, while 118 future-dependent labels change. This confirms that the features do not depend on future observations.

## Temporal splits

Rows are split by prediction date, and observations whose `target_date` crosses a split boundary are removed.

| Split | Dates | Rows | Tickers |
| --- | --- | ---: | ---: |
| Train | 2010–2014 | 583,707 | 485 |
| Validation | 2015 | 123,023 | 494 |
| Test | 2016 | 124,329 | 496 |

Positive-label rates are:

- Train: 51.90%
- Validation: 49.97%
- Test: 52.37%

The 2016 outcomes were not used for model or hyperparameter selection.

## Models and validation

The following models were evaluated:

- Majority-class baseline
- Logistic regression with standardized inputs and `C` in `{0.1, 1, 10}`
- Histogram gradient-boosted trees with:
  - `max_leaf_nodes` in `{15, 31}`
  - `learning_rate` in `{0.05, 0.1}`
  - 100 iterations
  - Internal early stopping disabled

All model choices used 2015 validation ROC-AUC. The classification threshold was frozen at 0.5.

Best validation ROC-AUC values were:

| Model family | Stock only | Stock plus sector | Sector improvement |
| --- | ---: | ---: | ---: |
| Logistic regression | 0.512619 | 0.524186 | +0.011567 |
| Boosted trees | 0.526532 | 0.530642 | +0.004111 |

The selected primary model is a histogram gradient-boosted classifier with:

- Stock and sector features
- 15 maximum leaf nodes
- Learning rate 0.05
- 100 iterations
- Threshold 0.5

Its validation ROC-AUC is 0.530642.

## Optional expanded model study

The notebook now contains an additional validation-only model spectrum. It is disabled by default so a normal notebook run does not train the computationally heavier models:

```python
RUN_EXPANDED_MODEL_STUDY = False
```

Set this flag to `True` only when intentionally running the optional study. It does not replace the frozen primary model and must not be used to retroactively tune against the already inspected 2016 test year.

The optional direction-classification comparison includes:

- Majority baseline
- Logistic regression
- Shrinkage linear discriminant analysis
- Extra Trees classifier
- Random Forest classifier
- Histogram gradient-boosted classifier
- Small, early-stopped multilayer perceptron

The optional continuous-return comparison predicts `next_return` with:

- Mean-return baseline
- Ordinary linear regression
- Ridge regression
- Elastic Net
- Huber robust regression
- Extra Trees regressor
- Random Forest regressor
- Histogram gradient-boosted regressor

Each family compares stock-only and stock-plus-sector features on identical rows. Regression metrics include MAE, RMSE, R-squared, Pearson correlation, Spearman rank correlation, directional accuracy, directional ROC-AUC, and a gross validation strategy return and Sharpe ratio.

The optional study also contains univariate ordinary-least-squares and standardized Ridge experiments for the 1-, 5-, and 20-session returns and both sector-relative returns. The sign of each fitted slope is used to describe momentum versus mean reversion.

The expanded model code is a post-hoc research extension. Because the 2016 outcomes were already inspected before it was added, its model choices should remain validation-only unless a genuinely new confirmation period is obtained.

## Held-out test results

After selection, the four chosen model configurations and the baseline were refit on the combined 2010–2015 rows and evaluated once on the same 2016 test rows.

| Model | Test ROC-AUC | Balanced accuracy | Accuracy |
| --- | ---: | ---: | ---: |
| Majority baseline | 0.500000 | 0.500000 | 0.523691 |
| Logistic, stock only | 0.509219 | 0.503803 | 0.520562 |
| Logistic, stock plus sector | 0.511411 | 0.502969 | 0.516830 |
| Boosted trees, stock only | 0.523533 | 0.513486 | 0.527005 |
| Boosted trees, stock plus sector | 0.523921 | 0.512234 | 0.526024 |

Sector features improved test ROC-AUC by:

- +0.002191 for logistic regression
- +0.000388 for boosted trees

The direction of the validation improvement carried into the test year, but its magnitude decreased substantially.

At the fixed threshold, the primary model predicted up on 79.2% of test rows. It identified approximately 80.3% of actual up sessions but only 22.1% of down or flat sessions.

## Sector analysis

The primary model's held-out sector ROC-AUC ranged from:

- 0.535538 for Consumer Staples
- 0.493566 for Real Estate

Nine of ten sectors had ROC-AUC above 0.5. The two largest sectors accounted for 30.8% of test stock-days, so aggregate performance was not supported by only one or two sectors, although it was uneven.

The notebook contains a sector results table and plots of sector ROC-AUC and support.

## Model interpretation

The notebook reports standardized logistic-regression coefficients and held-out permutation importance for the primary model.

The five-session stock return has the largest standardized logistic coefficient magnitude and the largest mean permutation importance. The five-session sector-relative return is also important, which is consistent with the feature-set experiment.

Permutation importance is treated as descriptive test analysis and is not used to alter the model.

## Classification-payoff illustration

An illustrative calculation divides $100 equally across all 124,329 test predictions and assumes an even-money payoff: one stake gained for a correct prediction and one stake lost for an incorrect prediction.

The primary model made:

- 65,400 correct predictions
- 58,929 incorrect predictions

Under those artificial assumptions:

- Profit: $5.20
- Return: 5.20%
- Final value: $105.20

This is a classification-payoff illustration, not a trading backtest, because it ignores actual return magnitudes, overlapping positions, costs, and execution.

## Return-based strategy simulation

A separate strategy layer uses actual `next_return` values and keeps the prediction model distinct from the trading policy.

Three daily equal-gross-exposure policies are evaluated:

1. Equal-weight always long
2. Model long-only, allocated across predicted-up stocks
3. Model long/short, long predicted-up stocks and short predicted-down stocks

Capital starts at $100, is rebalanced across available stocks, and compounds daily for 251 test sessions.

The simulation reports gross results and net results under a configurable 10-basis-point full daily round-trip cost.

| Strategy | Gross final value | Net final value |
| --- | ---: | ---: |
| Always long | $115.66 | $89.99 |
| Model long-only | $121.29 | $94.37 |
| Model long/short | $120.44 | $93.71 |

The long/short strategy has the strongest gross risk-adjusted profile:

- Gross annualized Sharpe: 1.876
- Gross maximum drawdown: -7.56%
- Gross final value: $120.44
- Net final value at the configured cost: $93.71

The model strategies outperform the always-long benchmark before costs, but all three lose money under the conservative full daily round-trip cost assumption. The economic result is therefore highly sensitive to turnover and execution costs.

The notebook includes accounting assertions, gross and net metrics, net equity curves, and a long/short drawdown plot.

## Important limitations

### Execution timing

The features use the completed close at session `t`, while `next_return` begins at that same close. A strategy generally cannot observe the completed close and then retroactively enter at that price.

The return simulation is therefore a research approximation unless signals can be formed before the close and executed in the closing auction. A deployable next-open strategy would require a new target and a complete rerun of feature selection, validation, and testing.

### Dataset scope

- The ticker and sector metadata are snapshots, not time-varying historical membership.
- Survivorship and coverage effects are not modeled.
- Conclusions are limited to the supplied tickers and dates.

### Trading realism

- The transaction-cost assumption is configurable but simplified.
- Slippage, bid-ask spreads, borrow availability, shorting costs, market impact, taxes, and capacity are not modeled directly.
- The analysis does not establish a deployable or generally profitable trading strategy.

## Reproducibility

From the project directory:

```bash
source .venv/bin/activate
jupyter notebook sp500_stock_direction.ipynb
```

To execute the complete notebook non-interactively:

```bash
.venv/bin/jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace sp500_stock_direction.ipynb \
  --ExecutePreprocessor.timeout=1200
```

The current notebook contains 55 cells, including 32 executed code cells. It validates successfully, all engineering assertions pass, and there are no saved execution errors or unfinished placeholders.

## Files created

- `sp500_stock_direction.ipynb` — complete analysis and saved outputs
- `requirements.txt` — pinned Python dependencies
- `.gitignore` — excludes the virtual environment, checkpoints, and raw data
- `data/README.md` — dataset source and download instructions
- `PROJECT_PROGRESS.md` — this project handoff document

