"""Illustrative ROS bag extraction example for corrfdd."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from corrfdd.ros_adapter import (  # noqa: E402
    BagExtractor,
    TopicMapping,
    extract_odometry,
    extract_twist,
    turtlebot3_mappings,
)


def main() -> None:
    # Option 1: Use built-in TurtleBot3 mappings.
    BagExtractor(turtlebot3_mappings())
    example_bag = Path("my_recording.bag")
    print(
        f"Use extractor.extract({example_bag!r}) to load a TurtleBot3 bag after installing corrfdd[ros]."
    )

    # Option 2: Define your own mappings for a custom robot.
    my_mappings = [
        TopicMapping(
            topic="/my_robot/odom",
            msg_type="nav_msgs/msg/Odometry",
            extractor=extract_odometry,
        ),
        TopicMapping(
            topic="/my_robot/cmd_vel",
            msg_type="geometry_msgs/msg/Twist",
            extractor=extract_twist,
        ),
    ]
    custom_extractor = BagExtractor(my_mappings)
    print(
        "Custom extractor configured for topics:",
        [mapping.topic for mapping in custom_extractor.topic_mappings],
    )


if __name__ == "__main__":
    main()
