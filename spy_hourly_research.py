"""Prespecified SPY improvements on cached bars; historical results are diagnostic."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from spy_hourly_audit import (
    AuditConfig,
    FEATURES,
    RAW_NAME,
    _fit,
    _label_deadline,
    _path_summary,
    _session_window,
    _training_rows,
    audit_fingerprint,
    cash_share_path,
    make_features,
)

TARGET = "return_oo"
NORMALIZED = ["ma_distance_pct", *FEATURES[1:]]
ACTIVITY = [
    "relative_range",
    "close_location",
    "previous_gap",
    "volume_surprise",
    "dollar_volume_surprise",
]
CLOCK = ["hour_sin", "hour_cos", "elapsed_hours_log"]


@dataclass(frozen=True)
class ResearchConfig:
    outer_start: str = "2020-01-01"
    diagnostic_start: str = "2025-01-15T20:00:00Z"
    fold_sessions: int = 63
    folds: int = 3
    regularization: tuple[float, ...] = (0.1, 1.0, 10.0)
    confidence_bands: tuple[float, ...] = (0.0, 0.025, 0.05)
    bootstrap_draws: int = 1000
    bootstrap_block_sessions: int = 20
    seed: int = 42


@dataclass(frozen=True)
class Stage:
    name: str
    normalized: bool = True
    stable: bool = True
    activity: bool = False
    clock: bool = False
    delay: bool = False

    @property
    def pools(self):
        core = NORMALIZED if self.normalized else FEATURES
        extras = (ACTIVITY if self.activity else []) + (CLOCK if self.clock else [])
        return {
            "indicators": core[:6] + extras,
            "lags": core[6:] + extras,
            "all": core + extras,
        }


STAGES = (
    Stage("A current", normalized=False, stable=False),
    Stage("B normalize only", stable=False),
    Stage("C stable normalized"),
    Stage("D clock", clock=True),
    Stage("E activity", activity=True),
    Stage("F combined", activity=True, clock=True),
    Stage("G extra-bar delay", delay=True),
)
PRIMARY = "C stable normalized"
METHOD = (
    "Holding-period target; quarterly expanding refits. A/B reproduce single "
    "126-session return selection. C–G use three preceding nonoverlapping "
    "63-session folds. Eligible candidates improve daily log loss against "
    "training-only class priors on average and in at least two folds, and "
    "beat buy-and-hold daily log returns in at least two folds. Among eligible "
    "candidates within one fold-based standard error of the best probability "
    "gain, prefer fewer columns then stronger regularization. Otherwise use "
    "buy-and-hold; confidence policy instead uses cash when no model qualifies. "
    "C is the prespecified primary procedure; D–G are sensitivities, not "
    "test-selected replacements. Primary costs are zero; 1bp is a sensitivity."
)


def study_fingerprint(data_dir="data", config=ResearchConfig()):
    return {
        "audit": audit_fingerprint(data_dir, AuditConfig()),
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "configuration": json.loads(json.dumps(asdict(config))),
    }


def research_features(bars, *, delay=False):
    out = make_features(bars, "sessions")
    out["ma_distance_pct"] = out["ma_distance"] / out["previous_close"]
    spread = bars.high - bars.low
    out["relative_range"] = (spread / bars.close).shift(1)
    out["close_location"] = (
        ((2 * bars.close - bars.high - bars.low) / spread.replace(0, np.nan))
        .where(spread.ne(0), 0.0)
        .shift(1)
    )
    out["previous_gap"] = np.log(bars.open / bars.close.shift(1)).shift(1)
    window, valid = _session_window(out.datetime, 1)
    for source, target in [
        ("volume", "volume_surprise"),
        ("dollar_volume", "dollar_volume_surprise"),
    ]:
        values = bars[source]
        if not np.isfinite(values).all() or values.lt(0).any():
            raise ValueError(f"{source} must be nonnegative and finite")
        logged = np.log1p(values)
        center = logged.rolling(window, min_periods=1).mean().where(valid)
        out[target] = (logged - center).shift(1)
    lagged = list(dict.fromkeys(FEATURES + NORMALIZED + ACTIVITY))
    if delay:
        # Only an extra completed bar of information; current clock is known.
        out[lagged] = out[lagged].shift(1)
    local = out.datetime.dt.tz_convert("America/New_York")
    angle = 2 * np.pi * (local.dt.hour + local.dt.minute / 60) / 24
    out["hour_sin"], out["hour_cos"] = np.sin(angle), np.cos(angle)
    elapsed = out.datetime.diff().dt.total_seconds() / 3600
    out["elapsed_hours_log"] = np.log1p(elapsed)
    out[lagged + CLOCK] = out[lagged + CLOCK].replace([np.inf, -np.inf], np.nan)
    return out


def probability_losses(y, probability):
    y, p = np.asarray(y, float), np.asarray(probability, float)
    if (
        y.shape != p.shape
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
    ):
        raise ValueError("Require aligned binary labels and probabilities in [0,1]")
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p)), (p - y) ** 2


def probability_summary(frame):
    labeled = frame.loc[frame[TARGET].notna()]
    y = labeled[TARGET].gt(0).astype(int).to_numpy()
    p, prior = labeled.probability.to_numpy(), labeled.prior.to_numpy()
    loss, brier = probability_losses(y, p)
    prior_loss, prior_brier = probability_losses(y, prior)
    return {
        "labeled_bars": len(y),
        "log_loss": loss.mean(),
        "prior_log_loss": prior_loss.mean(),
        "log_loss_gain": (prior_loss - loss).mean(),
        "brier": brier.mean(),
        "prior_brier": prior_brier.mean(),
        "brier_gain": (prior_brier - brier).mean(),
        "accuracy": ((p > 0.5) == y).mean(),
        "auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
    }


def positions(probability, policy, band=0.0):
    p = np.asarray(probability, float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid probabilities")
    if not 0 <= band < 0.5:
        raise ValueError("Invalid confidence band")
    if policy == "long_short":
        return np.where(p > 0.5, 1.0, -1.0)
    if policy == "long_cash":
        return (p > 0.5).astype(float)
    if policy == "confidence":
        return np.where(p > 0.5 + band, 1.0, np.where(p <= 0.5 - band, -1.0, 0.0))
    raise ValueError(f"Unknown policy: {policy}")


def _daily(values, times, *, aggregate="mean"):
    dates = pd.to_datetime(times, utc=True).dt.tz_convert("America/New_York").dt.date
    return pd.Series(np.asarray(values), index=dates).groupby(level=0).agg(aggregate)


def _folds(history, cutoff, stage, config):
    sessions = sorted(history.session.unique())
    count = config.folds if stage.stable else 1
    width = config.fold_sessions if stage.stable else 126
    if len(sessions) < count * width + 126:
        raise ValueError("Insufficient history for expanding inner folds")
    starts = [
        history.loc[history.session.eq(sessions[-offset * width]), "datetime"].min()
        for offset in range(count, 0, -1)
    ]
    for i, start in enumerate(starts):
        stop = starts[i + 1] if i + 1 < count else cutoff
        train = _training_rows(history, start, TARGET)
        validation = history.loc[history.datetime.ge(start) & history.datetime.lt(stop)]
        yield i, start, stop, train, validation


def _gain_record(prediction, stop):
    known = _training_rows(prediction, stop, TARGET)
    y = known[TARGET].gt(0).astype(int)
    loss, brier = probability_losses(y, known.probability)
    prior_loss, prior_brier = probability_losses(y, known.prior)
    gain = _daily(prior_loss - loss, known.datetime).mean()
    brier_gain = _daily(prior_brier - brier, known.datetime).mean()
    path = cash_share_path(prediction, positions(prediction.probability, "long_short"))
    hold = cash_share_path(prediction, np.ones(len(prediction)))
    excess = _daily(
        np.log1p(path.net_return) - np.log1p(hold.net_return),
        prediction.datetime,
        aggregate="sum",
    ).mean()
    return gain, brier_gain, excess, path.equity.iloc[-1] - 1


def _select(records, stage):
    scores = []
    for (pool, c), group in records.groupby(["pool", "C"], sort=False):
        gains = group.log_loss_gain.to_numpy()
        scores.append(
            {
                "pool": pool,
                "C": c,
                "columns": stage.pools[pool],
                "gain": gains.mean(),
                "se": gains.std(ddof=1) / np.sqrt(len(gains)) if len(gains) > 1 else 0,
                "eligible": gains.mean() > 0
                and (gains > 0).sum() >= 2
                and group.excess_daily_log_return.gt(0).sum() >= 2,
                "return": group.portfolio_return.mean(),
            }
        )
    if stage.stable:
        eligible = [s for s in scores if s["eligible"]]
        if not eligible:
            return None
        best = max(eligible, key=lambda s: s["gain"])
        near = [s for s in eligible if s["gain"] >= best["gain"] - best["se"]]
        return min(near, key=lambda s: (len(s["columns"]), s["C"], -s["gain"]))
    best = sorted(scores, key=lambda s: (-s["return"], len(s["columns"]), s["C"]))[0]
    if best["return"] <= records.hold_return.iloc[0] + 1e-12:
        return None
    return best


def _choose_band(validation, config):
    scores = []
    for band in config.confidence_bands:
        folds = []
        for _, fold in validation.groupby("fold"):
            path = cash_share_path(
                fold, positions(fold.probability, "confidence", band)
            )
            folds.append(
                _daily(np.log1p(path.net_return), fold.datetime, aggregate="sum").mean()
            )
        scores.append((np.mean(folds), band))
    # Fixed tie break: smaller band; outer returns never enter this decision.
    return max(scores, key=lambda item: (item[0], -item[1]))[1]


def _choose(history, cutoff, stage, config):
    records, predictions = [], {}
    for fold, start, stop, train, validation in _folds(history, cutoff, stage, config):
        prior = train[TARGET].gt(0).mean()
        hold_return = validation.close.iloc[-1] / validation.open.iloc[0] - 1
        for pool, columns in stage.pools.items():
            for c in config.regularization:
                model = _fit(train, columns, TARGET, c)
                prediction = validation.copy()
                prediction["probability"] = model.predict_proba(validation[columns])[
                    :, 1
                ]
                prediction["prior"], prediction["fold"] = prior, fold
                gain, brier, excess, ret = _gain_record(prediction, stop)
                records.append(
                    {
                        "fold": fold,
                        "pool": pool,
                        "C": c,
                        "log_loss_gain": gain,
                        "brier_gain": brier,
                        "excess_daily_log_return": excess,
                        "portfolio_return": ret,
                        "hold_return": hold_return,
                        "train_rows": len(train),
                        "last_training_label": _label_deadline(train, TARGET).max(),
                        "validation_start": start,
                        "validation_end": stop,
                        "last_scored_label": _label_deadline(
                            _training_rows(prediction, stop, TARGET), TARGET
                        ).max(),
                    }
                )
                predictions.setdefault((pool, c), []).append(prediction)
    records = pd.DataFrame(records)
    chosen = _select(records, stage)
    band = (
        _choose_band(pd.concat(predictions[(chosen["pool"], chosen["C"])]), config)
        if chosen is not None
        else None
    )
    return chosen, band, records


def _walk(frame, stage, config):
    columns = list(dict.fromkeys(sum(stage.pools.values(), [])))
    complete = frame.dropna(subset=columns)
    first = pd.Timestamp(config.outer_start, tz="America/New_York")
    expected = frame.loc[frame.datetime.ge(first), "datetime"].reset_index(drop=True)
    actual = complete.loc[complete.datetime.ge(first), "datetime"].reset_index(
        drop=True
    )
    if not expected.equals(actual):
        raise ValueError("A feature arm changed the outer evaluation timestamps")
    last = frame.datetime.max().tz_convert("America/New_York").tz_localize(None)
    parts, folds, decisions, coefficients = [], [], [], []
    for quarter in pd.period_range(first.tz_localize(None), last, freq="Q"):
        cutoff = quarter.start_time.tz_localize("America/New_York").tz_convert("UTC")
        end = (quarter + 1).start_time.tz_localize("America/New_York").tz_convert("UTC")
        history = complete.loc[complete.completion.lt(cutoff)]
        test = complete.loc[
            complete.datetime.ge(cutoff) & complete.datetime.lt(end)
        ].copy()
        if test.empty:
            continue
        chosen, band, record = _choose(history, cutoff, stage, config)
        record["quarter"], record["stage"] = str(quarter), stage.name
        folds.append(record)
        train = _training_rows(history, cutoff, TARGET)
        prior = train[TARGET].gt(0).mean()
        if chosen is None:
            test["probability"] = prior
            test["long_short"], test["long_cash"], test["confidence"] = 1.0, 1.0, 0.0
        else:
            model = _fit(train, chosen["columns"], TARGET, chosen["C"])
            test["probability"] = model.predict_proba(test[chosen["columns"]])[:, 1]
            for policy in ("long_short", "long_cash", "confidence"):
                test[policy] = positions(test.probability, policy, band)
            fitted = model.named_steps["logisticregression"]
            coefficients.extend(
                {
                    "stage": stage.name,
                    "quarter": str(quarter),
                    "feature": name,
                    "standardized_coefficient": value,
                }
                for name, value in zip(chosen["columns"], fitted.coef_[0])
            )
        test["prior"], test["quarter"] = prior, str(quarter)
        test["stage"], test["fallback"] = stage.name, chosen is None
        parts.append(test)
        decisions.append(
            {
                "stage": stage.name,
                "quarter": str(quarter),
                "choice": chosen["pool"] if chosen else "buy and hold",
                "C": chosen["C"] if chosen else None,
                "columns": ", ".join(chosen["columns"]) if chosen else "",
                "confidence_band": band,
                "training_prior": prior,
                "last_refit_label": _label_deadline(train, TARGET).max(),
                "refit_cutoff": cutoff,
                "outer_start": test.datetime.min(),
                "train_rows": len(train),
            }
        )
        print(f"{stage.name} {quarter}: {decisions[-1]['choice']}", flush=True)
    return (
        pd.concat(parts, ignore_index=True),
        pd.concat(folds, ignore_index=True),
        pd.DataFrame(decisions),
        pd.DataFrame(coefficients),
    )


def _daily_path(path):
    dates = (
        pd.to_datetime(path.datetime, utc=True)
        .dt.tz_convert("America/New_York")
        .dt.date
    )
    return (1 + path.set_index(dates).net_return).groupby(level=0).prod() - 1


def paired_risk(strategy, market, config):
    """Ex-post beta matching is descriptive, never an input to historical trades."""
    daily = pd.concat([strategy.rename("strategy"), market.rename("market")], axis=1)
    if daily.isna().any().any():
        raise ValueError("Risk comparisons require identical trading dates")
    y, x = daily.strategy.to_numpy(), daily.market.to_numpy()
    design = np.column_stack([np.ones(len(x)), x])
    alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]
    rng = np.random.default_rng(config.seed)
    length = min(config.bootstrap_block_sessions, len(x))
    boot = []
    for _ in range(config.bootstrap_draws):
        starts = rng.integers(
            0, len(x) - length + 1, size=int(np.ceil(len(x) / length))
        )
        indices = (starts[:, None] + np.arange(length)).ravel()[: len(x)]
        boot.append((y[indices] - x[indices]).mean())
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "beta": beta,
        "alpha_daily_bps": alpha * 10000,
        "excess_daily_bps": (y - x).mean() * 10000,
        "excess_daily_bps_ci_low": low * 10000,
        "excess_daily_bps_ci_high": high * 10000,
        "expost_beta_matched_market_return": np.prod(1 + beta * x) - 1,
    }


def run_study(
    data_dir="data", output_dir="artifacts/spy_hourly_research", config=ResearchConfig()
):
    fingerprint = study_fingerprint(data_dir, config)
    bars = pd.read_parquet(Path(data_dir) / "raw" / "SPY" / RAW_NAME)
    bars = bars.sort_values("datetime").reset_index(drop=True)
    frames = {delay: research_features(bars, delay=delay) for delay in (False, True)}
    predictions, fold_records, decisions, coefficients = [], [], [], []
    for stage in STAGES:
        prediction, folds, decision, coefficient = _walk(
            frames[stage.delay], stage, config
        )
        predictions.append(prediction)
        fold_records.append(folds)
        decisions.append(decision)
        coefficients.append(coefficient)
    predictions = pd.concat(predictions, ignore_index=True)
    fold_records = pd.concat(fold_records, ignore_index=True)
    decisions = pd.concat(decisions, ignore_index=True)
    coefficients = pd.concat(coefficients, ignore_index=True)
    baseline = predictions.loc[predictions.stage.eq(STAGES[0].name)].copy()
    paths, portfolio, scores, calibration, annual = [], [], [], [], []
    samples = [
        ("2020 onward", None),
        ("original test", pd.Timestamp(config.diagnostic_start)),
    ]
    # Identical entry and final liquidation conventions for every strategy in each span.
    for span, start in samples:
        market_frame = (
            baseline if start is None else baseline.loc[baseline.datetime.ge(start)]
        )
        market_path = cash_share_path(market_frame, np.ones(len(market_frame)))
        market_daily = _daily_path(market_path)
        portfolio.append(
            {
                "span": span,
                "stage": "buy and hold",
                "policy": "long_short",
                "cost_bps": 0.0,
                **_path_summary(market_path),
            }
        )
        for stage, prediction in predictions.groupby("stage", sort=False):
            frame = (
                prediction
                if start is None
                else prediction.loc[prediction.datetime.ge(start)]
            )
            scores.append({"span": span, "stage": stage, **probability_summary(frame)})
            if start is None:
                labeled = frame.loc[frame[TARGET].notna()]
                for source in ("probability", "prior"):
                    bins = np.minimum((10 * labeled[source]).astype(int), 9)
                    for bin_id, group in labeled.groupby(bins):
                        calibration.append(
                            {
                                "stage": stage,
                                "source": source,
                                "bin": bin_id,
                                "count": len(group),
                                "mean_probability": group[source].mean(),
                                "observed_frequency": group[TARGET].gt(0).mean(),
                            }
                        )
            policies = (
                ("long_short", "long_cash", "confidence")
                if stage in (STAGES[0].name, PRIMARY)
                else ("long_short",)
            )
            for policy in policies:
                for cost in (0.0, 1.0):
                    path = cash_share_path(frame, frame[policy], cost_bps=cost)
                    risk = (
                        paired_risk(_daily_path(path), market_daily, config)
                        if cost == 0
                        else {}
                    )
                    portfolio.append(
                        {
                            "span": span,
                            "stage": stage,
                            "policy": policy,
                            "cost_bps": cost,
                            **_path_summary(path),
                            "cash_rate": frame[policy].eq(0).mean(),
                            **risk,
                        }
                    )
                    if start is None:
                        for key, value in {
                            "stage": stage,
                            "policy": policy,
                            "cost_bps": cost,
                        }.items():
                            path[key] = value
                        paths.append(path)
                        if cost == 0:
                            daily = _daily_path(path)
                            for year, values in daily.groupby(
                                pd.DatetimeIndex(daily.index).year
                            ):
                                annual.append(
                                    {
                                        "stage": stage,
                                        "policy": policy,
                                        "year": year,
                                        "return": (1 + values).prod() - 1,
                                    }
                                )
    if study_fingerprint(data_dir, config) != fingerprint:
        raise ValueError("Study implementation or inputs changed during execution")
    for column, cutoff in [
        ("last_training_label", "validation_start"),
        ("last_scored_label", "validation_end"),
    ]:
        if not (
            pd.to_datetime(fold_records[column], utc=True)
            < pd.to_datetime(fold_records[cutoff], utc=True)
        ).all():
            raise ValueError("An inner fold used unavailable labels")
    if not (
        pd.to_datetime(decisions.last_refit_label, utc=True)
        < pd.to_datetime(decisions.refit_cutoff, utc=True)
    ).all():
        raise ValueError("An outer refit used unavailable labels")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prediction_columns = [
        "datetime",
        "stage",
        "quarter",
        "probability",
        "prior",
        TARGET,
        "next_open_time",
        "fallback",
        "long_short",
        "long_cash",
        "confidence",
    ]
    predictions[prediction_columns].to_csv(out / "probabilities.csv", index=False)
    fold_records.to_csv(out / "validation_folds.csv", index=False)
    decisions.to_csv(out / "decisions.csv", index=False)
    coefficients.to_csv(out / "coefficients.csv", index=False)
    pd.concat(paths, ignore_index=True).to_csv(out / "paths.csv", index=False)
    pd.DataFrame(portfolio).to_csv(out / "portfolio_results.csv", index=False)
    pd.DataFrame(scores).to_csv(out / "probability_scores.csv", index=False)
    pd.DataFrame(calibration).to_csv(out / "calibration.csv", index=False)
    pd.DataFrame(annual).to_csv(out / "annual_returns.csv", index=False)
    report = {
        "fingerprint": fingerprint,
        "method": METHOD,
        "primary": PRIMARY,
        "retrospective": True,
        "stages": [asdict(s) for s in STAGES],
        "outer_bars_per_stage": len(baseline),
        "portfolio": portfolio,
        "probability": scores,
        "fallback_quarters": decisions.loc[decisions.choice.eq("buy and hold")]
        .groupby("stage")
        .size()
        .to_dict(),
    }
    (out / "research_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    report = run_study()
    print(
        pd.DataFrame(report["portfolio"])[
            ["span", "stage", "policy", "cost_bps", "return", "daily_sharpe"]
        ].to_string(index=False)
    )
