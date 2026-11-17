"""Regression tests for the SPY notebook's timing and accounting repairs."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from spy_hourly_audit import (
    FEATURES,
    AuditConfig,
    _choose_on_past,
    _path_summary,
    _session_window,
    _training_rows,
    _walk_forward,
    cash_share_path,
    make_features,
)


def bars_fixture(days=30, start="2024-02-01"):
    dates = pd.bdate_range(start, periods=days)
    times = pd.DatetimeIndex(
        [
            day.tz_localize("America/New_York") + pd.Timedelta(hours=hour)
            for day in dates
            for hour in range(4, 20)
        ]
    ).tz_convert("UTC")
    close = 100 + np.arange(len(times)) * 0.01
    return pd.DataFrame(
        {
            "datetime": times,
            "open": close - 0.02,
            "high": close + 1,
            "low": close - 1,
            "close": close,
        }
    )


def ledger_bars(opens, closes, previous):
    return pd.DataFrame(
        {
            "datetime": pd.date_range(
                "2025-01-02 15:00", periods=len(opens), freq="h", tz="UTC"
            ),
            "open": opens,
            "close": closes,
            "previous_close": previous,
        }
    )


class FeatureWindowTests(unittest.TestCase):
    def test_legacy_formulas_match_direct_arithmetic(self):
        bars = bars_fixture()
        frame = make_features(bars, "legacy")
        position = 100
        previous = bars.close.iloc[position - 1]
        expected_ma = bars.close.iloc[position - 80 : position].mean() - previous
        past = bars.close.iloc[position - 8 : position].to_numpy()
        upper = past.mean() + 2 * past.std(ddof=0)
        self.assertAlmostEqual(frame.ma_distance.iloc[position], expected_ma)
        self.assertAlmostEqual(
            frame.upper_relative.iloc[position], (previous - upper) / upper
        )
        self.assertAlmostEqual(frame.pgo.iloc[position], (previous - past.mean()) / 2)

    def test_actual_session_window_across_weekend_and_dst(self):
        bars = bars_fixture(start="2024-03-01")
        local = bars.datetime.dt.tz_convert("America/New_York")
        monday_five = bars.index[
            (local.dt.date == pd.Timestamp("2024-03-11").date()) & local.dt.hour.eq(5)
        ][0]
        # Forecast at Monday 05:00 uses bars from Friday 05:00 through
        # Monday 04:00, exactly 16 observed bars despite the DST weekend.
        expected = bars.close.iloc[monday_five - 16 : monday_five].to_numpy()
        frame = make_features(bars, "sessions")
        upper = expected.mean() + 2 * expected.std(ddof=0)
        self.assertAlmostEqual(
            frame.upper_relative.iloc[monday_five], (expected[-1] - upper) / upper
        )

    def test_ten_sessions_are_not_eighty_bars(self):
        bars = bars_fixture()
        frame = make_features(bars, "sessions")
        position = 20 * 16 + 6
        expected = (
            bars.close.iloc[position - 160 : position].mean()
            - bars.close.iloc[position - 1]
        )
        self.assertAlmostEqual(frame.ma_distance.iloc[position], expected)
        self.assertNotAlmostEqual(
            frame.ma_distance.iloc[position],
            make_features(bars, "legacy").ma_distance.iloc[position],
        )

    def test_window_uses_available_bars_not_a_fixed_sixteen(self):
        bars = bars_fixture()
        bars = bars.drop(index=16 * 18 + 12).reset_index(drop=True)
        indexer, valid = _session_window(bars.datetime, 1)
        counts = np.arange(1, len(bars) + 1) - indexer.starts
        self.assertTrue((counts[valid] == 15).any())

    def test_current_and_future_prices_cannot_change_current_predictors(self):
        bars = bars_fixture()
        changed = bars.copy()
        changed.loc[250:, ["open", "high", "low", "close"]] *= 2
        for mode in ["legacy", "sessions"]:
            left = make_features(bars, mode)
            right = make_features(changed, mode)
            pd.testing.assert_frame_equal(
                left.loc[:250, FEATURES], right.loc[:250, FEATURES]
            )


class LedgerTests(unittest.TestCase):
    def test_fixed_share_short_does_not_earn_reciprocal_price_return(self):
        frame = ledger_bars([100, 90], [90, 81], [100, 90])
        path = cash_share_path(frame, [-1, -1])
        self.assertAlmostEqual(path.equity.iloc[-1], 1.19)
        self.assertAlmostEqual(path.dollar_turnover.iloc[1], 0.81)  # Final exit only.

    def test_gap_belongs_to_existing_position_not_the_new_prediction(self):
        frame = ledger_bars([100, 110], [100, 100], [100, 100])
        path = cash_share_path(frame, [1, -1])
        self.assertAlmostEqual(path.equity.iloc[-1], 1.2)

    def test_starting_flat_does_not_receive_pre_entry_gap(self):
        frame = ledger_bars([110], [100], [100])
        self.assertAlmostEqual(cash_share_path(frame, [1]).equity.iloc[-1], 100 / 110)
        self.assertAlmostEqual(
            cash_share_path(frame, [1], entry="previous_close").equity.iloc[-1], 1
        )

    def test_buy_hold_and_entry_exit_fees_match_closed_form(self):
        frame = ledger_bars([100, 110], [110, 120], [100, 110])
        path = cash_share_path(frame, [1, 1], cost_bps=1)
        self.assertAlmostEqual(path.equity.iloc[-1], 1.2 * (1 - 0.0001) / (1 + 0.0001))
        self.assertAlmostEqual(path.fees.sum(), 0.0001 * path.dollar_turnover.sum())

    def test_zero_cost_full_long_matches_price_ratio(self):
        bars = bars_fixture()
        bars["previous_close"] = bars.close.shift(1)
        path = cash_share_path(bars, np.ones(len(bars)))
        self.assertAlmostEqual(
            path.equity.iloc[-1], bars.close.iloc[-1] / bars.open.iloc[0]
        )

    def test_daily_risk_groups_by_new_york_session(self):
        frame = ledger_bars([100, 101], [101, 102], [100, 101])
        path = cash_share_path(frame, [1, 1])
        stats = _path_summary(path)
        self.assertEqual(stats["sessions"], 1)
        self.assertAlmostEqual(stats["return"], 0.02)


class ChronologyTests(unittest.TestCase):
    def test_overnight_label_waits_until_next_open_is_observed(self):
        frame = pd.DataFrame(
            {
                "completion": pd.to_datetime(
                    ["2024-12-31 22:00Z", "2025-01-02 10:00Z"]
                ),
                "next_open_time": pd.to_datetime(
                    ["2025-01-02 09:00Z", "2025-01-02 10:00Z"]
                ),
                "return_oc": [0.01, -0.01],
                "return_oo": [0.01, -0.01],
            }
        )
        cutoff = pd.Timestamp("2025-01-01", tz="UTC")
        self.assertEqual(_training_rows(frame, cutoff, "return_oc").index.tolist(), [0])
        self.assertTrue(_training_rows(frame, cutoff, "return_oo").empty)

    def test_inner_training_does_not_consume_validation_labels(self):
        frame = make_features(
            bars_fixture(days=160, start="2021-01-04"), "sessions"
        ).dropna(subset=FEATURES)
        observed = []

        def fake_fit(train, columns, target, c):
            observed.append(train.completion.max())
            return object()

        with (
            patch("spy_hourly_audit._fit", side_effect=fake_fit),
            patch(
                "spy_hourly_audit._predict",
                side_effect=lambda model, test, columns: np.ones(len(test)),
            ),
        ):
            chosen, last, boundary = _choose_on_past(
                frame, AuditConfig(inner_sessions=10)
            )
        self.assertLess(last, boundary)
        self.assertTrue(all(time < boundary for time in observed))
        self.assertEqual(chosen["name"], "buy and hold")  # Ties prefer the benchmark.

    def test_outer_refits_use_only_earlier_completed_labels(self):
        frame = make_features(bars_fixture(days=320, start="2020-01-02"), "sessions")
        chosen = {"name": "buy and hold", "columns": [], "c": None, "score": 0}

        def past_only(history, config):
            return chosen, history.completion.min(), history.datetime.max()

        with patch("spy_hourly_audit._choose_on_past", side_effect=past_only):
            _, audit = _walk_forward(frame, AuditConfig(outer_start="2021-01-01"))
        self.assertTrue(audit.last_refit_completion.lt(audit.outer_start).all())


if __name__ == "__main__":
    unittest.main()
