"""Tests for window segmentation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from corrfdd.exceptions import SegmentationError
from corrfdd.segmentation import (
    ACTION_IDLE,
    ACTION_MOVE_BACKWARD,
    ACTION_MOVE_FORWARD,
    ACTION_TURN,
    SegmenterConfig,
    WindowSegmenter,
    classify_action,
)


class TestClassifyAction:
    def test_move_forward(self) -> None:
        assert classify_action(0.5, 0.0) == ACTION_MOVE_FORWARD

    def test_move_backward(self) -> None:
        assert classify_action(-0.5, 0.0) == ACTION_MOVE_BACKWARD

    def test_idle(self) -> None:
        assert classify_action(0.0, 0.0) == ACTION_IDLE

    def test_turn(self) -> None:
        assert classify_action(0.0, 0.5) == ACTION_TURN

    def test_turn_takes_priority(self) -> None:
        assert classify_action(0.5, 0.5) == ACTION_TURN

    def test_custom_thresholds(self) -> None:
        config = SegmenterConfig(linear_threshold=0.1, angular_threshold=0.2)
        assert classify_action(0.08, 0.0, config) == ACTION_IDLE
        assert classify_action(0.15, 0.0, config) == ACTION_MOVE_FORWARD


class TestWindowSegmenter:
    def _make_csv(self, tmp_path: Path, actions: list[tuple[str, int]]) -> Path:
        rows = []
        timestamp = pd.Timestamp("2024-02-07 12:00:00")
        delta = pd.Timedelta("500ms")

        for action, sample_count in actions:
            for _ in range(sample_count):
                if action == ACTION_MOVE_FORWARD:
                    linear_x, angular_z = 0.5, 0.0
                elif action == ACTION_MOVE_BACKWARD:
                    linear_x, angular_z = -0.5, 0.0
                elif action == ACTION_TURN:
                    linear_x, angular_z = 0.0, 0.5
                else:
                    linear_x, angular_z = 0.0, 0.0

                rows.append(
                    {
                        "timestamp": timestamp,
                        "cmd_vel_linear_x": linear_x,
                        "cmd_vel_angular_z": angular_z,
                        "odom_x": 0.0,
                    }
                )
                timestamp += delta

        df = pd.DataFrame(rows).set_index("timestamp")
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path)
        return csv_path

    def test_segments_contiguous_actions(self, tmp_path: Path) -> None:
        csv_path = self._make_csv(
            tmp_path,
            [(ACTION_MOVE_FORWARD, 20), (ACTION_TURN, 10), (ACTION_MOVE_BACKWARD, 20)],
        )
        windows = WindowSegmenter().segment(csv_path)
        actions = [window.action for window in windows]
        assert ACTION_MOVE_FORWARD in actions
        assert ACTION_MOVE_BACKWARD in actions

    def test_short_windows_filtered(self, tmp_path: Path) -> None:
        config = SegmenterConfig(min_window_duration=5.0)
        csv_path = self._make_csv(tmp_path, [(ACTION_MOVE_FORWARD, 2)])
        windows = WindowSegmenter(config=config).segment(csv_path)
        assert len(windows) == 0

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "bad.csv"
        pd.DataFrame({"odom_x": [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3)).to_csv(
            csv_path
        )

        with pytest.raises(SegmentationError, match="Missing required"):
            WindowSegmenter().segment(csv_path)

    def test_nonexistent_csv_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SegmentationError, match="not found"):
            WindowSegmenter().segment(tmp_path / "missing.csv")

    def test_window_record_has_data(self, tmp_path: Path) -> None:
        csv_path = self._make_csv(tmp_path, [(ACTION_MOVE_FORWARD, 10)])
        windows = WindowSegmenter().segment(csv_path)
        for window in windows:
            assert len(window.data) > 0
            assert "cmd_vel_linear_x" in window.data.columns
