"""Independent oracles and causality checks for the SPY improvement study."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from spy_hourly_audit import _fit
from spy_hourly_research import (
    ACTIVITY,
    CLOCK,
    NORMALIZED,
    ResearchConfig,
    STAGES,
    TARGET,
    _choose_band,
    _folds,
    _gain_record,
    _select,
    _walk,
    paired_risk,
    positions,
    probability_losses,
    probability_summary,
    research_features,
)
from test_spy_hourly_audit import bars_fixture, ledger_bars


def activity_bars(days=30):
    bars = bars_fixture(days=days)
    bars["volume"] = 100 + np.arange(len(bars)) % 17
    bars["dollar_volume"] = bars.volume * bars.close
    return bars


class ResearchFeatureTests(unittest.TestCase):
    def test_normalized_ma_is_scale_invariant(self):
        bars = activity_bars()
        scaled = bars.copy()
        scaled[["open", "high", "low", "close"]] *= 7
        a, b = research_features(bars), research_features(scaled)
        np.testing.assert_allclose(a[NORMALIZED], b[NORMALIZED], atol=1e-9)
        self.assertAlmostEqual(
            a.ma_distance_pct.iloc[400],
            (bars.close.iloc[240:400].mean() - bars.close.iloc[399])
            / bars.close.iloc[399],
        )

    def test_activity_and_clock_have_analytic_values(self):
        bars = activity_bars()
        frame = research_features(bars)
        i = 400
        self.assertAlmostEqual(frame.relative_range.iloc[i], 2 / bars.close.iloc[i - 1])
        self.assertAlmostEqual(frame.close_location.iloc[i], 0)
        self.assertAlmostEqual(
            frame.previous_gap.iloc[i],
            np.log(bars.open.iloc[i - 1] / bars.close.iloc[i - 2]),
        )
        self.assertAlmostEqual(
            frame.volume_surprise.iloc[i],
            np.log1p(bars.volume.iloc[i - 1])
            - np.log1p(bars.volume.iloc[i - 16 : i]).mean(),
        )
        local = frame.datetime.iloc[i].tz_convert("America/New_York")
        self.assertAlmostEqual(
            frame.hour_sin.iloc[i], np.sin(2 * np.pi * local.hour / 24)
        )

    def test_future_mutation_does_not_change_any_predictor(self):
        bars = activity_bars()
        changed = bars.copy()
        changed.loc[
            250:, ["open", "high", "low", "close", "volume", "dollar_volume"]
        ] *= 3
        for delay in (False, True):
            a, b = (
                research_features(bars, delay=delay),
                research_features(changed, delay=delay),
            )
            pd.testing.assert_frame_equal(
                a.loc[:250, NORMALIZED + ACTIVITY + CLOCK],
                b.loc[:250, NORMALIZED + ACTIVITY + CLOCK],
            )

    def test_delay_uses_one_more_completed_bar_but_current_clock(self):
        bars = activity_bars()
        regular, delayed = research_features(bars), research_features(bars, delay=True)
        np.testing.assert_allclose(
            delayed[NORMALIZED + ACTIVITY].iloc[251],
            regular[NORMALIZED + ACTIVITY].iloc[250],
        )
        pd.testing.assert_frame_equal(regular[CLOCK], delayed[CLOCK])

    def test_zero_volume_and_zero_range_remain_finite_after_warmup(self):
        bars = activity_bars()
        bars["volume"], bars["dollar_volume"] = 0.0, 0.0
        bars.high, bars.low = bars.close, bars.close
        result = research_features(bars)
        self.assertTrue(np.isfinite(result[ACTIVITY].iloc[200:]).all().all())


class ProbabilityTests(unittest.TestCase):
    def test_scores_match_sklearn_and_ignore_unavailable_final_target(self):
        y, p = [0, 1, 0, 1], [0.1, 0.8, 0.4, 0.9]
        loss, brier = probability_losses(y, p)
        self.assertAlmostEqual(loss.mean(), log_loss(y, p))
        self.assertAlmostEqual(brier.mean(), brier_score_loss(y, p))
        frame = pd.DataFrame(
            {TARGET: [-1, 1, -1, 1, np.nan], "probability": [*p, 1], "prior": [0.5] * 5}
        )
        summary = probability_summary(frame)
        self.assertEqual(summary["labeled_bars"], 4)
        self.assertAlmostEqual(summary["prior_log_loss"], np.log(2))

    def test_bad_probabilities_are_rejected(self):
        for p in ([np.nan], [-0.1], [1.1]):
            with self.assertRaises(ValueError):
                probability_losses([1], p)

    def test_position_boundaries_and_cash_band(self):
        p = [0.4, 0.49, 0.5, 0.51, 0.6]
        np.testing.assert_equal(positions(p, "long_short"), [-1, -1, -1, 1, 1])
        np.testing.assert_equal(positions(p, "long_cash"), [0, 0, 0, 1, 1])
        np.testing.assert_equal(positions(p, "confidence", 0.05), [-1, 0, 0, 0, 1])


class SelectionTests(unittest.TestCase):
    def test_one_standard_error_prefers_smaller_pool(self):
        rows = []
        for pool, gains in [("all", [0.023, 0.021, 0.019]), ("lags", [0.0205] * 3)]:
            for gain in gains:
                rows.append(
                    {
                        "pool": pool,
                        "C": 0.1,
                        "log_loss_gain": gain,
                        "excess_daily_log_return": 0.001,
                        "portfolio_return": 0.1,
                    }
                )
        self.assertEqual(_select(pd.DataFrame(rows), STAGES[2])["pool"], "lags")

    def test_single_fold_winner_and_probability_only_winner_are_rejected(self):
        for gains, excess in [
            ([0.1, -0.01, -0.01], [0.01] * 3),
            ([0.01] * 3, [-0.01] * 3),
        ]:
            rows = pd.DataFrame(
                {
                    "pool": ["all"] * 3,
                    "C": [1] * 3,
                    "log_loss_gain": gains,
                    "excess_daily_log_return": excess,
                    "portfolio_return": [0.1] * 3,
                }
            )
            self.assertIsNone(_select(rows, STAGES[2]))

    def test_expanding_folds_are_disjoint_and_labels_purged(self):
        frame = research_features(activity_bars(days=400)).dropna(subset=NORMALIZED)
        cutoff = frame.completion.max() + pd.Timedelta(hours=1)
        config = replace(ResearchConfig(), fold_sessions=20)
        parts = list(_folds(frame, cutoff, STAGES[2], config))
        self.assertEqual(len(parts), 3)
        self.assertTrue(set(parts[0][4].index).isdisjoint(parts[1][4].index))
        self.assertTrue(set(parts[1][4].index).isdisjoint(parts[2][4].index))
        for _, start, _, train, validation in parts:
            self.assertTrue(train.next_open_time.lt(start).all())
            self.assertEqual(validation.session.nunique(), 20)
        self.assertLess(len(parts[0][3]), len(parts[2][3]))

    def test_validation_does_not_score_label_at_next_fold_open(self):
        frame = ledger_bars([100, 101], [101, 102], [99, 100])
        frame["next_open_time"] = [
            frame.datetime.iloc[1],
            frame.datetime.iloc[1] + pd.Timedelta(days=3),
        ]
        frame[TARGET] = [0.01, -0.02]
        frame["probability"], frame["prior"] = [0.9, 0.01], 0.5
        stop = frame.datetime.iloc[1] + pd.Timedelta(hours=1)
        gain, _, _, _ = _gain_record(frame, stop)
        self.assertAlmostEqual(gain, np.log(0.9 / 0.5))

    def test_confidence_band_is_chosen_from_validation_payoffs(self):
        frame = ledger_bars([100, 99], [99, 100], [100, 99])
        frame["probability"], frame["fold"] = [0.51, 0.49], 0
        self.assertEqual(_choose_band(frame, ResearchConfig()), 0.025)

    def test_scaler_never_uses_validation_extremes(self):
        train = pd.DataFrame({"feature": [-2, -1, 1, 2], TARGET: [-1, -1, 1, 1]})
        model = _fit(train, ["feature"], TARGET)
        model.predict_proba(pd.DataFrame({"feature": [1e9]}))
        self.assertAlmostEqual(model.named_steps["standardscaler"].mean_[0], 0)

    def test_fallback_prior_and_policies_use_past_only(self):
        frame = research_features(activity_bars(days=400))
        frame[TARGET] = np.where(frame.datetime.dt.year.eq(2024), -0.01, 0.01)
        config = replace(ResearchConfig(), outer_start="2025-01-01")

        def fallback(history, cutoff, stage, config):
            self.assertTrue(history.completion.lt(cutoff).all())
            return None, None, pd.DataFrame()

        with patch("spy_hourly_research._choose", side_effect=fallback):
            prediction, _, decision, _ = _walk(frame, STAGES[2], config)
        self.assertTrue(prediction.long_short.eq(1).all())
        self.assertTrue(prediction.confidence.eq(0).all())
        for quarter, group in prediction.groupby("quarter"):
            boundary = decision.loc[decision.quarter.eq(quarter), "refit_cutoff"].iloc[
                0
            ]
            eligible = frame.dropna(subset=NORMALIZED)
            expected = (
                eligible.loc[eligible.next_open_time.lt(boundary), TARGET].gt(0).mean()
            )
            self.assertAlmostEqual(group.prior.iloc[0], expected)


class RiskTests(unittest.TestCase):
    def test_analytic_alpha_beta_and_identical_strategy_bootstrap(self):
        market = pd.Series([-0.02, 0.01, -0.01, 0.03] * 20)
        config = replace(ResearchConfig(), bootstrap_draws=100)
        result = paired_risk(0.001 + 0.4 * market, market, config)
        self.assertAlmostEqual(result["beta"], 0.4)
        self.assertAlmostEqual(result["alpha_daily_bps"], 10)
        identical = paired_risk(market, market, config)
        self.assertEqual(identical["excess_daily_bps_ci_low"], 0)
        self.assertEqual(identical["excess_daily_bps_ci_high"], 0)


if __name__ == "__main__":
    unittest.main()
