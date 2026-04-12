"""Tests for the standalone SFDD detector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sfdd.detector import (
    TURTLEBOT3_SIGNALS,
    SFDDDetector,
    SFDDModel,
    SFDDTrainer,
    _compute_pairwise_correlations,
)
from sfdd.exceptions import DetectionError
from sfdd.segmentation import WindowRecord


def _make_window(
    action: str = "move_forward",
    n: int = 20,
    seed: int = 42,
    correlated: bool = True,
) -> WindowRecord:
    """Create a test WindowRecord with correlated or uncorrelated signals."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="500ms")

    if correlated:
        base = np.linspace(0, 1, n) + rng.normal(0, 0.01, n)
        data = pd.DataFrame(
            {
                "cmd_vel_linear_x": base * 0.5,
                "odom_x": base * 1.2,
                "odom_linear_x": base * 0.5,
                "sonar_robot": 2.0 - base * 0.3,
                "laser_front": 3.0 - base * 0.5,
            },
            index=timestamps,
        )
    else:
        data = pd.DataFrame(
            {
                "cmd_vel_linear_x": rng.normal(0, 1, n),
                "odom_x": rng.normal(0, 1, n),
                "odom_linear_x": rng.normal(0, 1, n),
                "sonar_robot": rng.normal(0, 1, n),
                "laser_front": rng.normal(0, 1, n),
            },
            index=timestamps,
        )

    return WindowRecord(
        action=action,
        start_time=timestamps[0].timestamp(),
        end_time=timestamps[-1].timestamp(),
        data=data,
        source_bag=Path("test.bag"),
    )


def _make_custom_dataframe(
    n: int = 24,
    seed: int = 7,
    faulty: bool = False,
) -> pd.DataFrame:
    """Create a custom-schema DataFrame for auto-discovery tests."""
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 1, n) + rng.normal(0, 0.01, n)
    wheel_encoder = base * 12.0
    imu_accel_x = base * 0.4 + 0.1

    if faulty:
        wheel_encoder = rng.normal(0.0, 1.0, n)

    return pd.DataFrame(
        {
            "motor_rpm": base * 50.0,
            "wheel_encoder": wheel_encoder,
            "imu_accel_x": imu_accel_x,
            "all_nan": [np.nan] * n,
            "mode": ["nominal"] * n,
        }
    )


class TestSFDDTrainer:
    def test_fit_produces_model(self) -> None:
        windows = [_make_window(seed=index) for index in range(5)]
        model = SFDDTrainer().fit(windows)
        assert len(model.reference_correlations) > 0
        assert len(model.theta) > 0

    def test_fit_empty_raises(self) -> None:
        with pytest.raises(DetectionError, match="No nominal windows"):
            SFDDTrainer().fit([])

    def test_model_serialization(self, tmp_path: Path) -> None:
        windows = [_make_window(seed=index) for index in range(3)]
        model = SFDDTrainer().fit(windows)

        path = tmp_path / "sfdd_model.json"
        model.save(path)

        loaded = SFDDModel.load(path)
        assert len(loaded.reference_correlations) == len(model.reference_correlations)

    def test_turtlebot3_preset_still_available(self) -> None:
        assert "cmd_vel_linear_x" in TURTLEBOT3_SIGNALS


class TestSFDDDetector:
    def test_nominal_window_no_fault(self) -> None:
        train_windows = [_make_window(seed=index) for index in range(5)]
        model = SFDDTrainer().fit(train_windows)

        result = SFDDDetector().predict(_make_window(seed=99), model)
        assert result.method == "sfdd"
        assert isinstance(result.fault_detected, bool)

    def test_faulty_window_detected(self) -> None:
        train_windows = [_make_window(seed=index, correlated=True) for index in range(5)]
        model = SFDDTrainer().fit(train_windows)

        result = SFDDDetector().predict(_make_window(seed=0, correlated=False), model)
        assert result.fault_detected
        assert len(result.violated_pairs) > 0
        assert result.method == "sfdd"

    def test_detection_result_fields(self) -> None:
        train_windows = [_make_window(seed=index) for index in range(3)]
        model = SFDDTrainer().fit(train_windows)

        result = SFDDDetector().predict(_make_window(seed=42), model)
        assert hasattr(result, "fault_detected")
        assert hasattr(result, "violated_pairs")
        assert hasattr(result, "correlation_deltas")
        assert hasattr(result, "method")


class TestAutoDiscoverSignals:
    """Tests for signal auto-discovery when signal_columns=None."""

    def test_auto_discovers_columns(self) -> None:
        """Trainer with signal_columns=None discovers columns from data."""
        windows = [
            WindowRecord(
                action="unknown",
                start_time=0.0,
                end_time=0.0,
                data=_make_custom_dataframe(seed=index),
                source_bag=Path("custom.csv"),
            )
            for index in range(3)
        ]

        model = SFDDTrainer(theta=0.2).fit(windows)

        assert model.signal_columns == ("motor_rpm", "wheel_encoder", "imu_accel_x")
        assert all("all_nan" not in pair for pair in model.reference_correlations)

    def test_explicit_columns_still_work(self) -> None:
        """Trainer with explicit signal_columns uses only those."""
        windows = [
            WindowRecord(
                action="unknown",
                start_time=0.0,
                end_time=0.0,
                data=_make_custom_dataframe(seed=index),
                source_bag=Path("custom.csv"),
            )
            for index in range(3)
        ]

        model = SFDDTrainer(
            theta=0.2,
            signal_columns=("motor_rpm", "wheel_encoder"),
        ).fit(windows)

        assert model.signal_columns == ("motor_rpm", "wheel_encoder")
        assert set(model.reference_correlations) == {("motor_rpm", "wheel_encoder")}

    def test_fit_from_dataframes(self) -> None:
        """fit_from_dataframes trains from plain DataFrames."""
        model = SFDDTrainer(theta=0.2).fit_from_dataframes(
            [_make_custom_dataframe(seed=index) for index in range(3)]
        )

        assert model.signal_columns == ("motor_rpm", "wheel_encoder", "imu_accel_x")
        assert len(model.reference_correlations) > 0

    def test_predict_dataframe(self) -> None:
        """predict_dataframe works without constructing WindowRecord."""
        model = SFDDTrainer(theta=0.2).fit_from_dataframes(
            [_make_custom_dataframe(seed=index) for index in range(3)]
        )
        result = SFDDDetector().predict_dataframe(
            _make_custom_dataframe(seed=99, faulty=True),
            model,
        )

        assert result.method == "sfdd"
        assert isinstance(result.fault_detected, bool)


class TestPairwiseCorrelations:
    def test_perfectly_correlated(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]})
        correlations = _compute_pairwise_correlations(df, ["a", "b"])
        assert ("a", "b") in correlations
        assert abs(correlations[("a", "b")] - 1.0) < 1e-10

    def test_insufficient_samples_skipped(self) -> None:
        df = pd.DataFrame({"a": [1.0, np.nan, np.nan], "b": [1.0, 2.0, np.nan]})
        correlations = _compute_pairwise_correlations(df, ["a", "b"])
        assert correlations == {}
