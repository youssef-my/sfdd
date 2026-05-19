"""Tests for optional ROS bag extraction helpers."""

from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from corrfdd.ros_adapter import (
    BagExtractor,
    TopicMapping,
    _check_rosbags_installed,
    extract_imu,
    extract_laser_scan,
    extract_odometry,
    extract_twist,
    turtlebot3_mappings,
)


class TestTopicMapping:
    def test_topic_mapping_construction(self) -> None:
        mapping = TopicMapping(
            topic="/cmd_vel",
            msg_type="geometry_msgs/msg/Twist",
            extractor=extract_twist,
        )
        assert mapping.topic == "/cmd_vel"
        assert mapping.msg_type == "geometry_msgs/msg/Twist"


class TestExtractors:
    def test_extract_twist(self) -> None:
        msg = SimpleNamespace(
            linear=SimpleNamespace(x=1.2),
            angular=SimpleNamespace(z=-0.3),
        )
        assert extract_twist(msg) == {
            "cmd_vel_linear_x": 1.2,
            "cmd_vel_angular_z": -0.3,
        }

    def test_extract_odometry(self) -> None:
        msg = SimpleNamespace(
            pose=SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=2.5, y=-1.5))),
            twist=SimpleNamespace(
                twist=SimpleNamespace(
                    linear=SimpleNamespace(x=0.4),
                    angular=SimpleNamespace(z=0.2),
                )
            ),
        )
        assert extract_odometry(msg) == {
            "odom_x": 2.5,
            "odom_y": -1.5,
            "odom_linear_x": 0.4,
            "odom_angular_z": 0.2,
        }

    def test_extract_laser_scan(self) -> None:
        msg = SimpleNamespace(ranges=[1.0, 2.0, 3.0, 4.0])
        assert extract_laser_scan(msg) == {
            "laser_0": 1.0,
            "laser_1": 2.0,
            "laser_2": 3.0,
            "laser_3": 4.0,
        }

    def test_extract_imu(self) -> None:
        msg = SimpleNamespace(
            linear_acceleration=SimpleNamespace(x=0.1, y=0.2, z=0.3),
            angular_velocity=SimpleNamespace(x=1.1, y=1.2, z=1.3),
        )
        assert extract_imu(msg) == {
            "imu_accel_x": 0.1,
            "imu_accel_y": 0.2,
            "imu_accel_z": 0.3,
            "imu_gyro_x": 1.1,
            "imu_gyro_y": 1.2,
            "imu_gyro_z": 1.3,
        }


class TestOptionalDependencyHandling:
    def test_module_import_is_lazy(self) -> None:
        assert BagExtractor.__name__ == "BagExtractor"

    def test_check_rosbags_installed_raises_clear_error(self) -> None:
        original_import = builtins.__import__

        def raising_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "rosbags":
                raise ImportError("missing rosbags")
            return original_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=raising_import),
            pytest.raises(
                ImportError,
                match="pip install rosbags",
            ),
        ):
            _check_rosbags_installed()

    def test_extract_raises_import_error_before_touching_bag(self) -> None:
        original_import = builtins.__import__

        def raising_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "rosbags":
                raise ImportError("missing rosbags")
            return original_import(name, *args, **kwargs)

        extractor = BagExtractor(topic_mappings=turtlebot3_mappings())
        with (
            patch("builtins.__import__", side_effect=raising_import),
            pytest.raises(
                ImportError,
                match="pip install rosbags",
            ),
        ):
            extractor.extract(Path("missing.bag"))

    def test_turtlebot3_mappings_returns_topic_mappings(self) -> None:
        mappings = turtlebot3_mappings()
        assert mappings
        assert all(isinstance(mapping, TopicMapping) for mapping in mappings)
