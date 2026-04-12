"""Minimal standalone example for training and running SFDD."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sfdd = importlib.import_module("sfdd")


def make_window(seed: int, faulty: bool = False) -> sfdd.WindowRecord:
    """Create a synthetic sensor window with optionally broken correlations."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=40, freq="250ms")
    base = np.linspace(0, 1, len(timestamps)) + rng.normal(0, 0.02, len(timestamps))

    odom_x = base * 1.3
    odom_linear_x = base * 0.6
    sonar_robot = 2.5 - base * 0.4
    laser_front = 3.0 - base * 0.6

    if faulty:
        odom_linear_x = rng.normal(0.0, 0.6, len(timestamps))

    data = pd.DataFrame(
        {
            "cmd_vel_linear_x": base * 0.6,
            "odom_x": odom_x,
            "odom_linear_x": odom_linear_x,
            "sonar_robot": sonar_robot,
            "laser_front": laser_front,
        },
        index=timestamps,
    )

    return sfdd.WindowRecord(
        action="move_forward",
        start_time=timestamps[0].timestamp(),
        end_time=timestamps[-1].timestamp(),
        data=data,
        source_bag=Path(f"synthetic_{seed}.csv"),
    )


def main() -> None:
    nominal_windows = [make_window(seed=index) for index in range(5)]
    model = sfdd.SFDDTrainer(theta=0.25).fit(nominal_windows)

    detector = sfdd.SFDDDetector()
    test_window = make_window(seed=99, faulty=True)
    result = detector.predict(test_window, model)

    print("Fault detected:", result.fault_detected)
    print("Violated pairs:", result.violated_pairs[:5])
    print("Tracked pairs:", len(result.correlation_deltas))


if __name__ == "__main__":
    main()
