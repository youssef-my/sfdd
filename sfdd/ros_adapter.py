"""Optional ROS bag extraction helpers for SFDD."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sfdd.segmentation import WindowRecord, WindowSegmenter

logger = logging.getLogger(__name__)

DEFAULT_LASER_SAMPLES = 8


def _check_rosbags_installed() -> None:
    try:
        import rosbags  # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'rosbags' package is required for ROS bag extraction. "
            "Install it with: pip install rosbags"
        ) from None


@dataclass(frozen=True)
class TopicMapping:
    """Maps a ROS topic to one or more signal column names.

    Attributes:
        topic: The ROS topic name (e.g. "/odom").
        msg_type: The ROS message type string (e.g. "nav_msgs/msg/Odometry").
        extractor: A callable that takes a deserialized ROS message
            and returns a dict of {column_name: float_value}.
    """

    topic: str
    msg_type: str
    extractor: Callable[[Any], dict[str, float]]


def extract_twist(msg: Any) -> dict[str, float]:
    """Extract a minimal command-velocity view from a Twist message."""
    return {
        "cmd_vel_linear_x": float(msg.linear.x),
        "cmd_vel_angular_z": float(msg.angular.z),
    }


def extract_odometry(msg: Any) -> dict[str, float]:
    """Extract position and velocity features from an Odometry message."""
    return {
        "odom_x": float(msg.pose.pose.position.x),
        "odom_y": float(msg.pose.pose.position.y),
        "odom_linear_x": float(msg.twist.twist.linear.x),
        "odom_angular_z": float(msg.twist.twist.angular.z),
    }


def extract_laser_scan(msg: Any) -> dict[str, float]:
    """Extract evenly spaced laser range samples from a LaserScan message."""
    ranges = np.asarray(getattr(msg, "ranges", []), dtype=float)
    if ranges.size == 0:
        return {}

    sample_count = min(DEFAULT_LASER_SAMPLES, ranges.size)
    indices = np.linspace(0, ranges.size - 1, num=sample_count, dtype=int)
    result: dict[str, float] = {}
    for output_index, range_index in enumerate(indices):
        value = ranges[range_index]
        result[f"laser_{output_index}"] = float(value) if np.isfinite(value) else float("nan")
    return result


def extract_imu(msg: Any) -> dict[str, float]:
    """Extract linear acceleration and angular velocity from an IMU message."""
    return {
        "imu_accel_x": float(msg.linear_acceleration.x),
        "imu_accel_y": float(msg.linear_acceleration.y),
        "imu_accel_z": float(msg.linear_acceleration.z),
        "imu_gyro_x": float(msg.angular_velocity.x),
        "imu_gyro_y": float(msg.angular_velocity.y),
        "imu_gyro_z": float(msg.angular_velocity.z),
    }


def turtlebot3_mappings() -> list[TopicMapping]:
    """Return default topic mappings for TurtleBot3."""
    return [
        TopicMapping(
            topic="/cmd_vel",
            msg_type="geometry_msgs/msg/Twist",
            extractor=extract_twist,
        ),
        TopicMapping(
            topic="/odom",
            msg_type="nav_msgs/msg/Odometry",
            extractor=extract_odometry,
        ),
        TopicMapping(
            topic="/scan",
            msg_type="sensor_msgs/msg/LaserScan",
            extractor=extract_laser_scan,
        ),
        TopicMapping(
            topic="/imu",
            msg_type="sensor_msgs/msg/Imu",
            extractor=extract_imu,
        ),
    ]


class BagExtractor:
    """Extract ROS1/ROS2 bag files into DataFrames for SFDD."""

    def __init__(
        self,
        topic_mappings: list[TopicMapping],
        resample_freq: str = "500ms",
    ) -> None:
        self.topic_mappings = topic_mappings
        self.resample_freq = resample_freq

    def extract(self, bag_path: Path) -> pd.DataFrame:
        """Extract a single bag file to a DataFrame."""
        _check_rosbags_installed()

        bag_path = Path(bag_path)
        if bag_path.is_dir():
            frames = self._extract_ros2_frames(bag_path)
        else:
            frames = self._extract_ros1_frames(bag_path)

        if not frames:
            raise ValueError(f"No data extracted from {bag_path}")

        combined = pd.concat(frames, axis=1).sort_index()
        combined = combined.groupby(level=0).last()
        return combined.resample(self.resample_freq).ffill().dropna(how="all")

    def extract_to_csv(self, bag_path: Path, output_path: Path) -> Path:
        """Extract a bag file and save as CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.extract(bag_path).to_csv(output_path)
        return output_path

    def extract_to_windows(
        self,
        bag_path: Path,
        segmenter: WindowSegmenter | None = None,
    ) -> list[WindowRecord]:
        """Extract a bag file and segment into action windows.

        Combines extraction + segmentation in one step.
        Requires cmd_vel columns to be present for action classification.
        """
        df = self.extract(bag_path)
        return (segmenter or WindowSegmenter()).segment_dataframe(df, source_path=Path(bag_path))

    def _extract_ros1_frames(self, bag_path: Path) -> list[pd.DataFrame]:
        """Extract topic frames from a ROS1 bag."""
        if not bag_path.exists():
            raise FileNotFoundError(f"ROS bag not found: {bag_path}")

        from rosbags.rosbag1 import Reader
        from rosbags.typesys import Stores, get_typestore

        typestore = get_typestore(Stores.ROS1_NOETIC)
        frames: list[pd.DataFrame] = []

        with Reader(bag_path) as reader:
            for mapping in self.topic_mappings:
                connections = [
                    connection
                    for connection in reader.connections
                    if connection.topic == mapping.topic and connection.msgtype == mapping.msg_type
                ]
                frames.extend(
                    self._frames_for_connections(
                        reader.messages(connections=connections),
                        mapping,
                        lambda rawdata, msg_type: typestore.deserialize_ros1(rawdata, msg_type),
                    )
                )

        return frames

    def _extract_ros2_frames(self, bag_path: Path) -> list[pd.DataFrame]:
        """Extract topic frames from a ROS2 bag directory."""
        try:
            from rosbags.rosbag2 import Reader as Reader2
        except ImportError:
            raise ImportError(
                "Installed 'rosbags' package does not provide ROS2 bag support."
            ) from None

        from rosbags.typesys import Stores, get_typestore

        typestore = get_typestore(Stores.LATEST)
        frames: list[pd.DataFrame] = []

        with Reader2(bag_path) as reader:
            for mapping in self.topic_mappings:
                connections = [
                    connection
                    for connection in reader.connections
                    if connection.topic == mapping.topic and connection.msgtype == mapping.msg_type
                ]
                frames.extend(
                    self._frames_for_connections(
                        reader.messages(connections=connections),
                        mapping,
                        lambda rawdata, msg_type: typestore.deserialize_cdr(rawdata, msg_type),
                    )
                )

        return frames

    def _frames_for_connections(
        self,
        messages: Any,
        mapping: TopicMapping,
        deserialize: Callable[[bytes, str], Any],
    ) -> list[pd.DataFrame]:
        """Convert a sequence of raw bag messages for one mapping into frames."""
        rows: list[dict[str, float]] = []
        timestamps: list[pd.Timestamp] = []

        for connection, timestamp, rawdata in messages:
            logger.debug("Extracting %s (%s)", connection.topic, connection.msgtype)
            msg = deserialize(rawdata, connection.msgtype)
            extracted = mapping.extractor(msg)
            if not extracted:
                continue
            rows.append(extracted)
            timestamps.append(pd.to_datetime(timestamp, unit="ns"))

        if not rows:
            return []

        frame = pd.DataFrame(rows, index=pd.DatetimeIndex(timestamps))
        return [frame]
