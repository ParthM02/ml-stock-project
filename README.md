# ml-stock-project

SPY hourly direction research using the existing 2016-06-28 through 2026-07-10
snapshot. The active implementation is
[spy_hourly_model_full_experiment.ipynb](spy_hourly_model_full_experiment.ipynb),
with the repair and walk-forward experiments in sections 13–14 implemented by
[spy_hourly_audit.py](spy_hourly_audit.py).
Sections 15–17 add the controlled improvement study in
[spy_hourly_research.py](spy_hourly_research.py): normalized features,
three-window selection, probability/calibration diagnostics, activity/clock
feature ablations, long/cash/confidence policies, and an extra-bar-delay check.

Create the project environment and install dependencies:

```sh
uv venv --python 3.12
uv pip install -r requirements.txt
uv run --no-sync jupyter notebook spy_hourly_model_full_experiment.ipynb
```

Run from the repository root. Sections 13–14 can run independently of the
historical cells, using the imports in section 13. Their base case uses zero
trading costs; 1 bp per traded dollar is an optional sensitivity. Sections 1–12
retain the original experiment and its saved results, including its historical
1-bp configuration. The active sklearn models run on CPU.

The audit reads these existing files without downloading data:

- `data/raw/SPY/SPY-bars-1h-2016-06-28_2026-07-10.parquet`
- `data/processed/spy_hourly_feature_target_data_2016-06-28_2026-07-10.parquet`
- `data/processed/logistic_feature_search.sqlite`

Regenerate the reports directly with:

```sh
uv run --no-sync python spy_hourly_audit.py
uv run --no-sync python spy_hourly_research.py
```

Generated tables, equity paths, cutoff audits and the fingerprinted report are
saved under ignored `artifacts/spy_hourly_audit/`. The notebook reuses the report
only when the implementation, inputs, configuration and recorded library
versions match. The historical download cell optionally needs `alpaca-py` and
credentials; neither is needed to use the supplied snapshot or run the audit.
The improvement study saves its own probabilities, candidate validation scores,
decisions, coefficients and results in `artifacts/spy_hourly_research/`. Section
15 includes all imports needed to execute sections 15–17 independently; stale
or absent results trigger regeneration. It does not overwrite the repair audit.

The audit isolates cash/share accounting, next-open fills, session-aware
features and label availability, then tests quarterly model selection using
only earlier observations. All results are retrospective: the exploratory
split already exposed the nominal final test period. See
[PROJECT_PROGRESS.md](PROJECT_PROGRESS.md) for measured results and limitations.
The improvement study reproduces the original holding-target equity curve
exactly, but its prespecified stable selector does not improve returns across
both reported periods. The strongest clock-feature result uses an active model
in just one quarter. No new arm is automatically promoted from these results.

Run the remaining test suite and code checks with:

```sh
uv run --no-sync python -m unittest discover -v
uv tool run ruff format --check spy_hourly_audit.py spy_hourly_research.py test_spy_hourly_audit.py test_spy_hourly_research.py
uv tool run ruff check spy_hourly_audit.py spy_hourly_research.py test_spy_hourly_audit.py test_spy_hourly_research.py
```
