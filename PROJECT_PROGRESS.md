# Project progress

The project now uses the SPY hourly notebook and `spy_hourly_audit.py`.
The earlier daily-stock notebook, its seven implementation scripts, four test
scripts and optional Metal requirements file have been removed. Dependencies
and run instructions now describe the SPY workflow. Existing data and ignored
generated artifacts were preserved; current reports are under
`artifacts/spy_hourly_audit/`.

Retirement removed 1,810 production Python lines and 879 test lines (2,689
total; zero added Python lines). The SPY notebook, implementation and tests
are byte-for-byte unchanged. `uv run --no-project python -m unittest discover
-v` passes all 14 remaining tests; Ruff format/lint, notebook schema validation,
and the saved audit fingerprint also pass. No references to the deleted modules
remain in the active source or documentation.

## SPY hourly notebook: measured repair audit

Investigated the newly pulled teammate notebook using only its existing raw
parquet, processed parquet and SQLite feature search. `spy_hourly_audit.py`
owns the controlled experiment, session-aware indicators, cash/share ledger
and nested chronological re-selection. Notebook sections 13–14 contain executed
tables and plots; the original code and saved numeric outputs remain available
as provenance. Corrected stale markdown claiming a 43.30% all-feature result:
the cached original one-bp test returns are 33.74% for the selected seven,
22.80% for all fourteen and 27.30% for buy-and-hold. The exploratory 80/20 split
had already exposed the final 15%, so this remains retrospective research.

All controlled variants use the same 5,941 bars from 2025-01-15 20:00 UTC to
2026-07-10 20:00 UTC. The frozen seven-feature architecture gives:

| Cumulative change | Zero-cost return | 1 bp per traded dollar |
| --- | ---: | ---: |
| Original signed-log accounting | 36.18% | 33.74% |
| Self-financing cash/share accounting | 35.91% | 33.46% |
| Next-open fills, unchanged forecasts | 32.88% | 30.48% |
| Strict training-label cutoff | 32.85% | 30.41% |
| Actual session windows, refitted close-to-close label | 48.69% | 45.49% |
| Open-to-following-open label, session windows | 26.99% | 24.25% |
| Buy-and-hold | 27.31% | 27.28% |

The source contains a median 16 bars per New York trading date, not eight.
Session windows anchor to the same local time one/ten observed sessions earlier,
including DST/weekends; original double-SMA PGO smoothing is preserved to
isolate the window repair. An intrabar-label arm and an extra-bar-delay
sensitivity are also saved, without choosing a winner from this test period.

The ledger trades only when the requested side changes, carries existing
shares across gaps before rebalancing, and liquidates at the final close.
Short P&L uses cash plus shares instead of reciprocal price returns. Exact
fees on traded notional are optional; the base case deducts zero. Sharpe uses
New York daily equity returns rather than an assumed eight-bar annualizer.
Open-to-following-open training labels become usable only after the next
observed open; inner and outer fits enforce availability strictly before
their validation/prediction cutoff. Fill prices still assume execution at the
observed open without spread, impact or financing deductions in the base case.

Quarterly nested selection compares three feature pools (six indicators,
eight lags, all fourteen) at C=0.1/1/10 with buy-and-hold, using only the last
126 past sessions for inner validation. It never imports the original
future-selected seven into early rolling years. From 2020 through July 2026,
holding-label selection returns 162.07% versus buy-and-hold's 133.51% at zero
cost; daily Sharpe is 0.847 versus 0.754, drawdown -33.06% versus -35.00%.
With a one-bp sensitivity, selection returns 93.88% versus 133.46%.
On the original 2025–2026 test dates it returns only 10.93% versus 27.31% at
zero cost. The intrabar-label rolling variant returns 42.10% for the full
period and 21.13% on the original test dates. These conflicting periods and
the holding-label strategy's 95.8% long exposure do not establish robust alpha.
No model is promoted based on these diagnostics.

Fourteen regression tests pass: analytic indicator parity, sessions/DST/missing
hours, future-price mutation, fixed-share short compounding, gap ownership,
flat entry, exact entry/exit fees, buy-and-hold parity, daily risk grouping,
and strict inner/outer/next-open label availability. Validation commands:

```sh
uv run --no-project python -m unittest -v test_spy_hourly_audit
uv tool run ruff format --check spy_hourly_audit.py test_spy_hourly_audit.py
uv tool run ruff check spy_hourly_audit.py test_spy_hourly_audit.py
```

The full audit completed and all three input SHA256 hashes were unchanged.
Cached report reuse requires matching source/input/config/runtime fingerprints.
Generated reports, paths and cutoff audits live in ignored
`artifacts/spy_hourly_audit/`; the appended notebook sections need no Polars
installation or download cells. Notebook validation and new saved outputs pass.
Original SPY backtest cells are retained because exact reproduction and
repaired execution have different contracts.

## Controlled probability, selection, feature and policy improvements

`spy_hourly_research.py` reuses the canonical feature clocks, logistic fitting
and cash/share ledger from the repair audit. Notebook sections 15–17 execute
the new study independently. The target is open-to-following-observed-open
direction; all seven arms use the same 25,923 outer bars from 2020 onward and
the same 5,941-bar original test window. Neither cached data nor earlier audit
results were modified. The baseline reproduces every position and equity value
of the saved holding-target walk-forward run, within 1e-12 numerical tolerance.

The seven arms were defined before their new outer results were inspected:
A current selector; B normalize only; C stable normalized selector (the
prespecified primary proposed procedure); D C plus clock; E C plus activity;
F C plus both; G C with an extra completed bar of signal delay. They are
comparisons of procedures, not a test-based search for which arm to promote.

Normalization replaces dollar SMA distance with `(SMA10 - close) / close`,
using completed-bar values. Activity adds lagged relative high-low range,
closing location, opening gap, and volume/dollar-volume surprises (log1p
minus a one-session mean). Clock features are current New York hour sine and
cosine plus log1p hours since the previous observed bar. No new data is needed.
G delays all price/activity predictors by an additional bar while retaining
the current clock. All current/future OHLC and activity mutations leave current
predictors unchanged; normalized price features pass price-scale invariance.

A/B retain single-window 126-session return selection. C–G use three preceding
nonoverlapping 63-session validation folds with expanding earlier fits.
Candidates are indicators/lags/all at C=0.1/1/10. Eligibility requires positive
mean daily log-loss improvement over each fold's training-only class prior,
positive improvement in at least two folds, and positive daily log-return
excess over buy-and-hold in at least two folds. Among qualifying candidates
within one fold-based standard error of the best probability gain, choose the
fewest features, then stronger regularization. Otherwise use buy-and-hold.
Validation probability scores exclude labels unavailable before the fold ends;
every fit also enforces strict label availability before its start.

At zero costs, long/short results are:

| Arm | 2020–July 2026 return | Original 2025–2026 test return | Buy-and-hold fallback quarters |
| --- | ---: | ---: | ---: |
| A current | 162.07% | 10.93% | 9/27 |
| B normalize only | 163.67% | 4.89% | 7/27 |
| C stable normalized | 104.79% | 19.20% | 19/27 |
| D clock | 209.93% | 27.31% | 26/27 |
| E activity | 124.17% | 25.25% | 20/27 |
| F combined | 120.29% | 21.23% | 22/27 |
| G extra-bar delay | 85.87% | 28.98% | 20/27 |
| Buy and hold | 133.51% | 27.31% | — |

D's only active quarter is 2020Q1; its large full-span number is not evidence
of consistent forecasting improvement. C improves the recent window over A,
but loses the wider comparison against both A and buy-and-hold. Its full-period
log-loss gain is -0.000737 (worse than the prior baseline), with recent-window
gain +0.000120. F's full-period gain is only +0.000039; all other arms have
negative full-period gains. These magnitudes and mixed periods do not establish
robust predictive skill. No thresholds were retuned after these results.

Probability outputs include selected forecasts and training priors, log loss,
Brier, accuracy, AUC, and fixed-bin reliability tables with counts. Fallback
quarters use prior probabilities for diagnostics rather than fabricated 100%
confidence. The last unknown next-open label is omitted from forecast scoring.
No post-hoc probability calibrator is fitted. Pooled AUC includes variation of
the training prior between quarters, including when no active model is used.

A and C also compare long/cash at 0.5 with a long/cash/short confidence policy.
Confidence half-width is chosen from 0/0.025/0.05 on earlier validation daily
log returns only; ties prefer the smaller width. When no model qualifies,
confidence uses cash while the binary policies use buy-and-hold. This makes
exposure differences explicit. C long/cash returns 121.61% over the full span
and 23.23% on the recent window; C confidence returns -9.43% and -2.11%.
These policies are reported separately, without selecting the best outer curve.

Risk diagnostics report regression beta/intercept, an ex-post beta-matched
SPY benchmark and paired mean-daily-excess-return intervals from 1,000 moving
block bootstrap draws with 20 New York sessions per block. C long/short has
beta 0.753 and a full-period excess-return interval of [-2.382, 1.173] bps/day;
D's interval is [-0.454, 5.429] bps/day. Both include zero. The ex-post benchmark
is descriptive rather than a tradable fitted-in-advance strategy. Intervals
condition on existing predictions, do not refit the models, and do not adjust
for testing seven arms. Zero costs remain primary; one-bp sensitivities are
reported separately. Observed-open execution remains idealized.

All 30 tests pass (14 existing plus 16 new). The new tests cover independent
sklearn score parity, analytic feature/risk formulas, normalization, causal
mutation, extra delay, zero activity/range, policy boundaries, disjoint folds,
label purging at validation ends, one-standard-error selection, rejection of
inconsistent candidates, validation-only thresholds, training-only scaling,
and fallback priors. Run commands:

```sh
uv run --no-project python spy_hourly_research.py
uv run --no-project python -m unittest discover -v
uv tool run ruff check spy_hourly_audit.py spy_hourly_research.py test_spy_hourly_audit.py test_spy_hourly_research.py
```

The isolated study adds 597 implementation lines and 236 test lines. The
previous audit's code, public behavior and saved results remain intact.
Generated probabilities, candidate folds, cutoffs, coefficients, paths,
calibration bins, annual returns and a fingerprinted report are under ignored
`artifacts/spy_hourly_research/`. The implemented diagnostics are useful even
though the prespecified selection changes did not produce a consistent
economic improvement; this is an inconclusive historical research result.
