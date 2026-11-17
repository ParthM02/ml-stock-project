"""Controlled repairs of the cached SPY hourly notebook; no data downloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api.indexers import BaseIndexer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sklearn
from threadpoolctl import threadpool_limits

RAW_NAME = "SPY-bars-1h-2016-06-28_2026-07-10.parquet"
PROCESSED_NAME = "spy_hourly_feature_target_data_2016-06-28_2026-07-10.parquet"
LEGACY_COLUMNS = [
    "price_delta_w_ma_10d",
    "close_relative_to_upper_band",
    "close_relative_to_lower_band",
    "PGO_8",
    "log_return_ma_1d_lag_1",
    "log_return_ma_10d_lag_1",
    *[f"log_return_lag_{lag}" for lag in range(1, 9)],
]
FEATURES = [
    "ma_distance",
    "upper_relative",
    "lower_relative",
    "pgo",
    "mean_return_1_session",
    "mean_return_10_sessions",
    *[f"return_lag_{lag}" for lag in range(1, 9)],
]
FEATURE_NAMES = dict(zip(LEGACY_COLUMNS, FEATURES))


@dataclass(frozen=True)
class AuditConfig:
    cost_bps: float = 0.0
    sensitivity_cost_bps: float = 1.0
    outer_start: str = "2020-01-01"
    inner_sessions: int = 126
    regularization: tuple[float, ...] = (0.1, 1.0, 10.0)
    threads: int = 4
    walk_target: Literal["return_oc", "return_oo"] = "return_oc"


def audit_fingerprint(data_dir="data", config=AuditConfig()):
    root = Path(data_dir)
    files = [
        root / "raw" / "SPY" / RAW_NAME,
        root / "processed" / PROCESSED_NAME,
        root / "processed" / "logistic_feature_search.sqlite",
    ]
    return {
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "inputs": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
        "configuration": json.loads(json.dumps(asdict(config))),
        "runtime": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
    }


class _SessionWindow(BaseIndexer):
    def __init__(self, starts):
        super().__init__(window_size=0)
        self.starts = np.asarray(starts, dtype=np.int64)

    def get_window_bounds(
        self, num_values=0, min_periods=None, center=None, closed=None, step=None
    ):
        return self.starts, np.arange(num_values, dtype=np.int64) + 1


def _session_window(times, sessions):
    """End at each completed bar; anchor N NY sessions earlier at the same clock.

    Example: a bar ending Tuesday at 11:00 has a one-session window starting
    Monday at 11:00. A weekend/DST change is not counted as extra sessions.
    The returned rolling values are shifted once before becoming predictors.
    """
    utc = pd.DatetimeIndex(times).tz_convert("UTC").as_unit("ns")
    local = utc.tz_convert("America/New_York")
    dates = pd.Index(sorted(set(local.date)))
    codes = dates.get_indexer(local.date)
    bar_end = local + pd.Timedelta(hours=1)
    anchors = []
    valid = codes >= sessions
    for code, end in zip(codes, bar_end):
        day = dates[max(0, code - sessions)]
        anchor = pd.Timestamp(day).tz_localize("America/New_York") + pd.Timedelta(
            hours=end.hour, minutes=end.minute, seconds=end.second
        )
        anchors.append(anchor.tz_convert("UTC").value)
    anchors = np.asarray(anchors, dtype=np.int64)
    valid &= anchors >= utc.asi8[0]
    starts = np.searchsorted(utc.asi8, anchors, side="left")
    starts = np.minimum(starts, np.arange(len(utc)))
    return _SessionWindow(starts), valid


def make_features(bars, mode: Literal["legacy", "sessions"]):
    """Preserve indicator formulas; change only their window definition."""
    out = bars.copy().reset_index(drop=True)
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True)
    if (
        out["datetime"].duplicated().any()
        or not out["datetime"].is_monotonic_increasing
    ):
        raise ValueError("Bars must have unique, increasing timestamps")
    if (
        not np.isfinite(out[["open", "high", "low", "close"]].to_numpy()).all()
        or out[["open", "high", "low", "close"]].le(0).any().any()
    ):
        raise ValueError("OHLC prices must be positive and finite")
    close = out["close"]
    out["previous_close"] = close.shift(1)
    out["return_cc"] = np.log(close / close.shift(1))
    out["return_oc"] = np.log(close / out["open"])
    out["return_oo"] = np.log(out["open"].shift(-1) / out["open"])
    out["next_open_time"] = out["datetime"].shift(-1)
    out["completion"] = out["datetime"] + pd.Timedelta(hours=1)
    out["session"] = out["datetime"].dt.tz_convert("America/New_York").dt.date
    true_range = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - close.shift(1)).abs(),
            (out["low"] - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    if mode == "legacy":

        def rolling(series, n, statistic="mean"):
            return getattr(series.rolling(n * 8, min_periods=n * 8), statistic)(
                **({"ddof": 0} if statistic == "std" else {})
            )
    elif mode == "sessions":
        windows = {n: _session_window(out["datetime"], n) for n in (1, 10)}

        def rolling(series, n, statistic="mean"):
            indexer, valid = windows[n]
            group = series.rolling(indexer, min_periods=1)
            result = getattr(group, statistic)(
                **({"ddof": 0} if statistic == "std" else {})
            )
            complete = (
                group.count().to_numpy() == np.arange(1, len(out) + 1) - indexer.starts
            )
            return result.where(valid & complete)
    else:
        raise ValueError(f"Unknown feature-window mode: {mode}")
    ma1 = rolling(close, 1)
    sd1 = rolling(close, 1, "std")
    upper, lower = ma1 + 2 * sd1, ma1 - 2 * sd1
    out["ma_distance"] = (rolling(close, 10) - close).shift(1)
    out["upper_relative"] = ((close - upper) / upper.where(upper.gt(0))).shift(1)
    out["lower_relative"] = ((close - lower) / lower.where(lower.gt(0))).shift(1)
    # Preserve the teammate's double-SMA true-range PGO variant for attribution.
    out["pgo"] = (
        (close - ma1) / rolling(rolling(true_range, 1), 1).replace(0, np.nan)
    ).shift(1)
    out["mean_return_1_session"] = rolling(out["return_cc"], 1).shift(1)
    out["mean_return_10_sessions"] = rolling(out["return_cc"], 10).shift(1)
    for lag in range(1, 9):
        out[f"return_lag_{lag}"] = out["return_cc"].shift(lag)
    out[FEATURES] = out[FEATURES].replace([np.inf, -np.inf], np.nan)
    return out


def cash_share_path(frame, sides, *, cost_bps=0.0, entry="open", liquidate=True):
    """Self-financing fixed-share holdings, traded only when the side changes.

    Existing shares earn gaps before a next-open trade. On a reversal, solve
    post-fee equity exactly, then establish +/-100% of that equity in SPY.
    The short's share count is NOT automatically rebalanced each bar for free.
    """
    side = np.asarray(sides, dtype=float)
    if len(frame) != len(side) or len(side) == 0 or not np.isin(side, [-1, 0, 1]).all():
        raise ValueError("One position in {-1,0,1} is required for every bar")
    if entry not in {"open", "previous_close"} or not 0 <= cost_bps < 10000:
        raise ValueError("Invalid execution price or cost")
    execution = frame[entry].to_numpy(dtype=float)
    closes = frame["close"].to_numpy(dtype=float)
    if (
        not np.isfinite(execution).all()
        or not np.isfinite(closes).all()
        or (execution <= 0).any()
        or (closes <= 0).any()
    ):
        raise ValueError("Execution and close prices must be positive and finite")
    fee_rate = cost_bps / 10000
    cash, shares, previous_equity = 1.0, 0.0, 1.0
    equity, returns, fees, turnover, exposure = [], [], [], [], []
    for price, close, desired in zip(execution, closes, side):
        paid, traded = 0.0, 0.0
        at_entry = cash + shares * price
        if at_entry <= 0:
            raise ValueError("Portfolio equity exhausted before entry")
        if desired != np.sign(shares):
            if desired == 0:
                traded = abs(shares) * price
                paid = fee_rate * traded
                cash, shares = at_entry - paid, 0.0
            else:
                after_fee = (at_entry - fee_rate * abs(shares) * price) / (1 + fee_rate)
                if after_fee <= 0:
                    raise ValueError("Portfolio equity exhausted by trading costs")
                new_shares = desired * after_fee / price
                traded = abs(new_shares - shares) * price
                paid = fee_rate * traded
                cash = after_fee - new_shares * price
                shares = new_shares
        marked = cash + shares * close
        if marked <= 0:
            raise ValueError("Portfolio equity exhausted at close")
        exposure.append(abs(shares) * close / marked)
        equity.append(marked)
        returns.append(marked / previous_equity - 1)
        fees.append(paid)
        turnover.append(traded)
        previous_equity = marked
    if liquidate and shares != 0:
        exit_notional = abs(shares) * closes[-1]
        exit_fee = fee_rate * exit_notional
        equity[-1] -= exit_fee
        if equity[-1] <= 0:
            raise ValueError("Portfolio equity exhausted by liquidation costs")
        fees[-1] += exit_fee
        turnover[-1] += exit_notional
        prior = equity[-2] if len(equity) > 1 else 1.0
        returns[-1] = equity[-1] / prior - 1
    return pd.DataFrame(
        {
            "datetime": frame["datetime"].to_numpy(),
            "equity": equity,
            "net_return": returns,
            "side": side,
            "fees": fees,
            "dollar_turnover": turnover,
            "gross_exposure": exposure,
        }
    )


def _path_summary(path):
    local_dates = (
        pd.to_datetime(path["datetime"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.date
    )
    daily = (1 + path["net_return"]).groupby(local_dates).prod() - 1
    equity = path["equity"].to_numpy()
    peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    return {
        "return": equity[-1] - 1,
        "daily_sharpe": daily.mean() / daily.std() * np.sqrt(252)
        if daily.std() > 0
        else np.nan,
        "max_drawdown": np.min(equity / peak - 1),
        "long_rate": path["side"].gt(0).mean(),
        "bars": len(path),
        "sessions": len(daily),
        "fees_per_initial_dollar": path["fees"].sum(),
        "max_gross_exposure": path["gross_exposure"].max(),
    }


def _fit(frame, columns, target, c=1.0):
    if frame.empty or frame[target].gt(0).nunique() != 2:
        raise ValueError("Training requires observations of both target classes")
    model = make_pipeline(
        StandardScaler(), LogisticRegression(C=c, max_iter=1000, random_state=42)
    )
    with threadpool_limits(limits=4):
        model.fit(frame[columns], frame[target].gt(0).astype(np.int8))
    return model


def _predict(model, frame, columns):
    with threadpool_limits(limits=4):
        return 2 * model.predict(frame[columns]).astype(float) - 1


def _load_cached(data_dir):
    root = Path(data_dir)
    raw_path = root / "raw" / "SPY" / RAW_NAME
    processed_path = root / "processed" / PROCESSED_NAME
    db_path = root / "processed" / "logistic_feature_search.sqlite"
    bars = pd.read_parquet(raw_path).sort_values("datetime").reset_index(drop=True)
    legacy = make_features(bars, "legacy").dropna(subset=FEATURES + ["return_cc"])
    cached = pd.read_parquet(processed_path)
    if len(cached) != len(legacy):
        raise ValueError(
            "Raw reconstruction and processed cache have different row counts"
        )
    np.testing.assert_allclose(
        legacy[FEATURES], cached[LEGACY_COLUMNS], atol=1e-8, rtol=1e-8
    )
    np.testing.assert_allclose(legacy["return_cc"], cached["log_return"], atol=1e-14)
    x = cached[LEGACY_COLUMNS].to_numpy(dtype=np.float64)
    y = cached["log_return"].to_numpy(dtype=np.float64)
    h = hashlib.sha256()
    h.update(x.tobytes())
    h.update(y.tobytes())
    a, b = int(len(cached) * 0.7), int(len(cached) * 0.85)
    config = json.dumps(
        {
            "features": LEGACY_COLUMNS,
            "cost_bps": 1.0,
            "train_end": a,
            "validation_end": b,
            "data_hash": h.hexdigest(),
        },
        sort_keys=True,
    )
    search_id = hashlib.sha256(config.encode()).hexdigest()
    with sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True) as db:
        winner = db.execute(
            "SELECT features_json FROM feature_search_results WHERE search_id=? ORDER BY net_return DESC LIMIT 1",
            (search_id,),
        ).fetchone()
    if winner is None:
        raise ValueError("No checkpoint matches the exact cached data/configuration")
    selected = [FEATURE_NAMES[name] for name in json.loads(winner[0])]
    # Use the original exact values for the reproduction, avoiding float-library
    # differences in rolling formulas; rebuilt timestamps were verified above.
    legacy.loc[:, FEATURES] = cached[LEGACY_COLUMNS].to_numpy()
    legacy.loc[:, "return_cc"] = y
    return (
        bars,
        legacy,
        selected,
        legacy["datetime"].iloc[a],
        legacy["datetime"].iloc[b],
        search_id,
    )


def _training_rows(frame, cutoff, target):
    available = (
        frame["next_open_time"] if target == "return_oo" else frame["completion"]
    )
    return frame.loc[available.lt(cutoff) & frame[target].notna()]


def _label_deadline(frame, target):
    return frame["next_open_time"] if target == "return_oo" else frame["completion"]


def _choose_on_past(history, config):
    dates = sorted(history["session"].unique())
    if len(dates) <= config.inner_sessions + 126:
        raise ValueError("Insufficient history for inner training and validation")
    validation_start = history.loc[
        history["session"].eq(dates[-config.inner_sessions]), "datetime"
    ].min()
    train = _training_rows(history, validation_start, config.walk_target)
    validation = history.loc[history["datetime"].ge(validation_start)]
    hold_path = cash_share_path(
        validation, np.ones(len(validation)), cost_bps=config.cost_bps
    )
    best = {
        "name": "buy and hold",
        "columns": [],
        "c": None,
        "score": hold_path.equity.iloc[-1] - 1,
    }
    candidates = []
    for name, columns in [
        ("indicators", FEATURES[:6]),
        ("lags", FEATURES[6:]),
        ("all", FEATURES),
    ]:
        for c in config.regularization:
            fitted = _fit(train, columns, config.walk_target, c)
            side = _predict(fitted, validation, columns)
            score = (
                cash_share_path(validation, side, cost_bps=config.cost_bps).equity.iloc[
                    -1
                ]
                - 1
            )
            candidates.append((score, len(columns), c, name, columns))
    for score, _, c, name, columns in sorted(
        candidates, key=lambda item: (-item[0], item[1], item[2])
    ):
        if score > best["score"] + 1e-12:
            best = {"name": name, "columns": columns, "c": c, "score": score}
    return best, _label_deadline(train, config.walk_target).max(), validation_start


def _walk_forward(frame, config):
    parts, audits = [], []
    first = pd.Timestamp(config.outer_start, tz="America/New_York")
    last = frame["datetime"].max().tz_convert("America/New_York")
    quarters = pd.period_range(
        first.tz_localize(None), last.tz_localize(None), freq="Q"
    )
    complete = frame.dropna(subset=FEATURES)
    expected = frame.loc[frame["datetime"].ge(first), "datetime"]
    actual = complete.loc[complete["datetime"].ge(first), "datetime"]
    if not expected.reset_index(drop=True).equals(actual.reset_index(drop=True)):
        raise ValueError(
            "Missing outer-period features require an explicit hold policy"
        )
    for quarter in quarters:
        cutoff = quarter.start_time.tz_localize("America/New_York").tz_convert("UTC")
        end = (quarter + 1).start_time.tz_localize("America/New_York").tz_convert("UTC")
        history = complete.loc[complete["completion"].lt(cutoff)]
        test = complete.loc[
            complete["datetime"].ge(cutoff) & complete["datetime"].lt(end)
        ].copy()
        if test.empty:
            continue
        chosen, inner_last, validation_start = _choose_on_past(history, config)
        refit = _training_rows(history, cutoff, config.walk_target)
        if chosen["columns"]:
            model = _fit(refit, chosen["columns"], config.walk_target, chosen["c"])
            test["side"] = _predict(model, test, chosen["columns"])
        else:
            test["side"] = 1.0
        parts.append(test)
        audits.append(
            {
                "quarter": str(quarter),
                "choice": chosen["name"],
                "C": chosen["c"],
                "columns": ", ".join(chosen["columns"]),
                "validation_return": chosen["score"],
                "last_inner_training_completion": inner_last,
                "validation_start": validation_start,
                "last_refit_completion": _label_deadline(
                    refit, config.walk_target
                ).max(),
                "outer_start": test["datetime"].min(),
                "target": config.walk_target,
            }
        )
        print(
            f"{quarter} [{config.walk_target}]: chose {chosen['name']} C={chosen['c']} using past validation only",
            flush=True,
        )
    return pd.concat(parts, ignore_index=True), pd.DataFrame(audits)


def run_audit(
    data_dir="data", output_dir="artifacts/spy_hourly_audit", config=AuditConfig()
):
    fingerprint = audit_fingerprint(data_dir, config)
    bars, legacy, selected, validation_start, test_start, search_id = _load_cached(
        data_dir
    )
    session = make_features(bars, "sessions")
    delayed = session.copy()
    delayed[FEATURES] = delayed[FEATURES].shift(1)
    legacy_test = legacy.loc[legacy["datetime"].ge(test_start)]
    times = legacy_test["datetime"]
    records, paths = [], []
    costs = tuple(dict.fromkeys([config.cost_bps, config.sensitivity_cost_bps]))

    def record(stage, model_name, test, side, entry="open", legacy_log=False):
        for cost in costs:
            if legacy_log:
                turnover = np.abs(np.diff(side, prepend=0.0))
                log_return = (
                    side * test["return_cc"].to_numpy() - turnover * cost / 10000
                )
                wealth_before = np.r_[1.0, np.exp(np.cumsum(log_return))[:-1]]
                gross_end = wealth_before * np.exp(side * test["return_cc"].to_numpy())
                path = pd.DataFrame(
                    {
                        "datetime": test["datetime"].to_numpy(),
                        "equity": np.exp(np.cumsum(log_return)),
                        "net_return": np.expm1(log_return),
                        "side": side,
                        "fees": gross_end * -np.expm1(-turnover * cost / 10000),
                        "gross_exposure": np.abs(side),
                    }
                )
            else:
                path = cash_share_path(test, side, cost_bps=cost, entry=entry)
            stats = _path_summary(path)
            records.append(
                {"stage": stage, "model": model_name, "cost_bps": cost, **stats}
            )
            path["stage"], path["model"], path["cost_bps"] = stage, model_name, cost
            paths.append(path)

    for name, columns in [("selected 7", selected), ("all 14", FEATURES)]:
        model = _fit(
            legacy.loc[legacy["datetime"].lt(test_start)], columns, "return_cc"
        )
        side = _predict(model, legacy_test, columns)
        record("A original log accounting", name, legacy_test, side, legacy_log=True)
        record(
            "B cash/share accounting", name, legacy_test, side, entry="previous_close"
        )
        record("C next-open fills; same forecasts", name, legacy_test, side)
    for stage, source, target in [
        ("C2 strict cutoff; legacy windows", legacy, "return_cc"),
        ("D session windows + strict label cutoff", session, "return_cc"),
        ("E intrabar target + session windows", session, "return_oc"),
        ("E2 next-open holding-period target", session, "return_oo"),
        ("F extra observed-bar signal delay", delayed, "return_oc"),
    ]:
        complete = source.dropna(subset=FEATURES)
        train = _training_rows(complete, test_start, target)
        test = complete.loc[complete["datetime"].isin(times)]
        if (
            not test["datetime"]
            .reset_index(drop=True)
            .equals(times.reset_index(drop=True))
        ):
            raise ValueError("A repair changed the comparison's test timestamps")
        for name, columns in [("selected 7", selected), ("all 14", FEATURES)]:
            model = _fit(train, columns, target)
            record(stage, name, test, _predict(model, test, columns))
    for stage in [
        "A original log accounting",
        "B cash/share accounting",
        "C next-open fills; same forecasts",
        "C2 strict cutoff; legacy windows",
        "D session windows + strict label cutoff",
        "E intrabar target + session windows",
        "E2 next-open holding-period target",
        "F extra observed-bar signal delay",
    ]:
        record(
            stage,
            "buy and hold",
            legacy_test,
            np.ones(len(legacy_test)),
            entry="previous_close" if stage.startswith(("A", "B")) else "open",
            legacy_log=stage.startswith("A"),
        )
    walk, intrabar_audit = _walk_forward(
        session, replace(config, walk_target="return_oc")
    )
    walk_test = walk.loc[walk["datetime"].isin(times)]
    if len(walk_test) != len(legacy_test):
        raise ValueError("Walk-forward predictions do not cover the full original test")
    record(
        "G nested quarterly selection",
        "past-only selection",
        walk_test,
        walk_test["side"].to_numpy(),
    )
    record(
        "G nested quarterly selection",
        "buy and hold",
        walk_test,
        np.ones(len(walk_test)),
    )
    holding_walk, holding_audit = _walk_forward(
        session, replace(config, walk_target="return_oo")
    )
    holding_test = holding_walk.loc[holding_walk["datetime"].isin(times)]
    if len(holding_test) != len(legacy_test):
        raise ValueError(
            "Holding-period predictions do not cover the full original test"
        )
    record(
        "H nested selection; holding-period target",
        "past-only selection",
        holding_test,
        holding_test["side"].to_numpy(),
    )
    record(
        "H nested selection; holding-period target",
        "buy and hold",
        holding_test,
        np.ones(len(holding_test)),
    )
    audit = pd.concat([intrabar_audit, holding_audit], ignore_index=True)
    full_records, full_paths = [], []
    for cost in costs:
        for name, frame, sides in [
            ("intrabar-label selection", walk, walk["side"]),
            ("holding-label selection", holding_walk, holding_walk["side"]),
            ("buy and hold", walk, np.ones(len(walk))),
        ]:
            path = cash_share_path(frame, sides, cost_bps=cost)
            full_records.append(
                {"model": name, "cost_bps": cost, **_path_summary(path)}
            )
            path["model"], path["cost_bps"] = name, cost
            full_paths.append(path)
    all_paths = pd.concat(paths, ignore_index=True)
    summary = pd.DataFrame(records)
    all_full_paths = pd.concat(full_paths, ignore_index=True)
    counts = bars.groupby(
        pd.to_datetime(bars["datetime"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.date
    ).size()
    out = Path(output_dir)
    if audit_fingerprint(data_dir, config) != fingerprint:
        raise ValueError("Audit code, runtime or input data changed during execution")
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "controlled_results.csv", index=False)
    all_paths.to_csv(out / "controlled_paths.csv", index=False)
    pd.DataFrame(full_records).to_csv(out / "walk_forward_results.csv", index=False)
    all_full_paths.to_csv(out / "walk_forward_paths.csv", index=False)
    audit.to_csv(out / "walk_forward_audit.csv", index=False)
    report = {
        "fingerprint": fingerprint,
        "original_search_id": search_id,
        "original_selected_features": selected,
        "validation_boundary": str(validation_start),
        "test_start": str(test_start),
        "test_end": str(legacy_test["datetime"].iloc[-1]),
        "test_bars": len(legacy_test),
        "median_bars_per_ny_session": float(counts.median()),
        "mean_bars_per_ny_session": float(counts.mean()),
        "cost_base_bps": config.cost_bps,
        "cost_sensitivity_bps": config.sensitivity_cost_bps,
        "controlled": summary.to_dict("records"),
        "walk_forward": full_records,
        "retrospective": True,
    }
    (out / "audit_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    report = run_audit()
    print(
        pd.DataFrame(report["controlled"])[
            ["stage", "model", "cost_bps", "return", "daily_sharpe"]
        ].to_string(index=False)
    )
