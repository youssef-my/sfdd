"""Action-based window segmentation for correlation-detector sensor windows."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from corrfdd.exceptions import SegmentationError

logger = logging.getLogger(__name__)

LINEAR_THRESHOLD = 0.05
ANGULAR_THRESHOLD = 0.1
MIN_WINDOW_DURATION = 0.5

ACTION_MOVE_FORWARD = "move_forward"
ACTION_MOVE_BACKWARD = "move_backward"
ACTION_IDLE = "idle"
ACTION_TURN = "turn"


@dataclass(frozen=True)
class WindowRecord:
    """A single time window of sensor data for one action."""

    action: str
    start_time: float
    end_time: float
    data: pd.DataFrame
    source_bag: Path

    class Config:
        """Allow arbitrary types for frozen dataclass fields."""

        arbitrary_types_allowed = True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WindowRecord):
            return NotImplemented
        return (
            self.action == other.action
            and self.start_time == other.start_time
            and self.end_time == other.end_time
            and self.source_bag == other.source_bag
        )

    def __hash__(self) -> int:
        return hash((self.action, self.start_time, self.end_time, self.source_bag))

    @property
    def duration(self) -> float:
        """Duration of the window in seconds."""
        return self.end_time - self.start_time


@dataclass(frozen=True)
class SegmenterConfig:
    """Configuration for action classification and windowing."""

    linear_threshold: float = LINEAR_THRESHOLD
    angular_threshold: float = ANGULAR_THRESHOLD
    min_window_duration: float = MIN_WINDOW_DURATION


def classify_action(
    linear_x: float,
    angular_z: float,
    config: SegmenterConfig | None = None,
) -> str:
    """Classify the current action from cmd_vel values."""
    if config is None:
        config = SegmenterConfig()

    abs_angular = abs(angular_z)
    abs_linear = abs(linear_x)

    if abs_angular >= config.angular_threshold:
        return ACTION_TURN
    if linear_x > config.linear_threshold and abs_angular < config.angular_threshold:
        return ACTION_MOVE_FORWARD
    if linear_x < -config.linear_threshold and abs_angular < config.angular_threshold:
        return ACTION_MOVE_BACKWARD
    if abs_linear < config.linear_threshold and abs_angular < config.angular_threshold:
        return ACTION_IDLE

    return ACTION_IDLE


class WindowSegmenter:
    """Segment resampled sensor data into contiguous action windows."""

    def __init__(self, config: SegmenterConfig | None = None) -> None:
        self.config = config or SegmenterConfig()

    def segment(self, csv_path: Path) -> list[WindowRecord]:
        """Segment a resampled CSV into action windows."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise SegmentationError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        return self.segment_dataframe(df, source_path=csv_path)

    def segment_dataframe(
        self,
        df: pd.DataFrame,
        source_path: Path | None = None,
    ) -> list[WindowRecord]:
        """Segment a DataFrame into action windows."""
        required_cols = {"cmd_vel_linear_x", "cmd_vel_angular_z"}
        missing = required_cols - set(df.columns)
        if missing:
            raise SegmentationError(f"Missing required columns: {missing}")

        source = source_path or Path("unknown")
        actions = df.apply(
            lambda row: classify_action(
                row["cmd_vel_linear_x"],
                row["cmd_vel_angular_z"],
                self.config,
            ),
            axis=1,
        )

        windows: list[WindowRecord] = []
        action_changes = actions != actions.shift()
        group_ids = action_changes.cumsum()

        for _group_id, group_df in df.groupby(group_ids):
            action = actions.loc[group_df.index[0]]

            try:
                idx = group_df.index
                if hasattr(idx, "to_series"):
                    epoch = idx.to_series().apply(lambda x: x.timestamp())
                    start_time = float(epoch.iloc[0])
                    end_time = float(epoch.iloc[-1])
                else:
                    start_time = float(idx[0])
                    end_time = float(idx[-1])
            except (TypeError, ValueError, AttributeError):
                start_time = 0.0
                end_time = 0.0

            duration = end_time - start_time
            if duration < self.config.min_window_duration:
                logger.debug(
                    "Skipping short %s window (%.3fs < %.3fs)",
                    action,
                    duration,
                    self.config.min_window_duration,
                )
                continue

            windows.append(
                WindowRecord(
                    action=action,
                    start_time=start_time,
                    end_time=end_time,
                    data=group_df.copy(),
                    source_bag=source,
                )
            )

        logger.info(
            "Segmented %d windows from %s (actions: %s)",
            len(windows),
            source,
            {window.action for window in windows},
        )
        return windows
