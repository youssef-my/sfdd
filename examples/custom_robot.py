"""Example showing SFDD training on a custom robot signal schema."""

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


def make_robot_dataframe(seed: int, faulty: bool = False, length: int = 32) -> pd.DataFrame:
    """Create a custom robot recording with non-TurtleBot signal names."""
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 1, length) + rng.normal(0, 0.01, length)
    motor_right = base * 18.0

    if faulty:
        motor_right = rng.normal(0.0, 1.0, length)

    return pd.DataFrame(
        {
            "motor_left_rpm": base * 17.0,
            "motor_right_rpm": motor_right,
            "imu_pitch": base * 0.4,
            "distance_front": 2.5 - base * 0.3,
            "distance_rear": 1.8 + base * 0.2,
        }
    )


def main() -> None:
    trainer = sfdd.SFDDTrainer(theta=0.2)
    model = trainer.fit_from_dataframes([make_robot_dataframe(seed=index) for index in range(4)])
    detector = sfdd.SFDDDetector()

    result = detector.predict_dataframe(make_robot_dataframe(seed=101, faulty=True), model)
    print("Signal columns:", model.signal_columns)
    print("Fault detected:", result.fault_detected)
    print("Violated pairs:", result.violated_pairs[:5])


if __name__ == "__main__":
    main()
