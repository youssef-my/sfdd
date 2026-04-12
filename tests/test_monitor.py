"""Tests for the SFDD streaming monitor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sfdd import SFDDMonitor
from sfdd.detector import SFDDTrainer


def _make_stream_dataframe(
    n: int = 24,
    seed: int = 5,
    faulty: bool = False,
) -> pd.DataFrame:
    """Create a numeric stream window with strong pairwise structure."""
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 1, n) + rng.normal(0, 0.01, n)
    right_rpm = base * 12.0

    if faulty:
        right_rpm = rng.normal(0.0, 1.0, n)

    return pd.DataFrame(
        {
            "motor_left_rpm": base * 10.0,
            "motor_right_rpm": right_rpm,
            "imu_pitch": base * 0.2,
        }
    )


class TestSFDDMonitor:
    def test_no_result_until_window_full(self) -> None:
        """Monitor returns empty list until window_size readings received."""
        model = SFDDTrainer().fit_from_dataframes(
            [_make_stream_dataframe(seed=index) for index in range(3)]
        )
        monitor = SFDDMonitor(model, window_size=4, step_size=1)

        for index in range(3):
            result = monitor.update(_make_stream_dataframe(seed=99).iloc[index].to_dict())
            assert result == []

    def test_detects_fault_in_stream(self) -> None:
        """Monitor flags fault when correlated signals break."""
        model = SFDDTrainer(theta=0.2).fit_from_dataframes(
            [_make_stream_dataframe(seed=index) for index in range(3)]
        )
        monitor = SFDDMonitor(model, window_size=8, step_size=1)

        nominal = _make_stream_dataframe(seed=10)
        faulty = _make_stream_dataframe(seed=10, faulty=True)

        for _, row in nominal.iloc[:8].iterrows():
            monitor.update(row.to_dict())

        detection_results = []
        for _, row in faulty.iterrows():
            detection_results.extend(monitor.update(row.to_dict()))
            if detection_results and detection_results[-1].fault_detected:
                break

        assert detection_results
        assert detection_results[-1].fault_detected

    def test_step_size_controls_frequency(self) -> None:
        """With step_size=4, detection runs every 4th reading."""
        model = SFDDTrainer().fit_from_dataframes(
            [_make_stream_dataframe(seed=index) for index in range(3)]
        )
        monitor = SFDDMonitor(model, window_size=4, step_size=4)

        readings = _make_stream_dataframe(seed=11)
        result_counts = []
        for _, row in readings.iloc[:12].iterrows():
            result_counts.append(len(monitor.update(row.to_dict())))

        assert sum(result_counts) == 3

    def test_reset_clears_buffer(self) -> None:
        """After reset, buffer_size is 0 and needs full window again."""
        model = SFDDTrainer().fit_from_dataframes(
            [_make_stream_dataframe(seed=index) for index in range(3)]
        )
        monitor = SFDDMonitor(model, window_size=4, step_size=1)

        for _, row in _make_stream_dataframe(seed=12).iloc[:4].iterrows():
            monitor.update(row.to_dict())

        monitor.reset()

        assert monitor.buffer_size == 0
        assert monitor.update(_make_stream_dataframe(seed=12).iloc[0].to_dict()) == []

    def test_custom_signal_names(self) -> None:
        """Monitor works with arbitrary signal names (not TurtleBot3)."""
        model = SFDDTrainer().fit_from_dataframes(
            [_make_stream_dataframe(seed=index) for index in range(3)]
        )
        monitor = SFDDMonitor(model, window_size=4, step_size=1)

        results = []
        for _, row in _make_stream_dataframe(seed=13).iloc[:4].iterrows():
            results = monitor.update(row.to_dict())

        assert model.signal_columns == ("motor_left_rpm", "motor_right_rpm", "imu_pitch")
        assert len(results) == 1
