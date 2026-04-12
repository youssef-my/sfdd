"""Streaming SFDD example with a rolling live monitor."""

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


def make_nominal_dataframe(seed: int, length: int = 24) -> pd.DataFrame:
    """Create correlated nominal sensor data for training."""
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 1, length) + rng.normal(0, 0.01, length)
    return pd.DataFrame(
        {
            "motor_left_rpm": base * 10.0,
            "motor_right_rpm": base * 10.5,
            "imu_pitch": base * 0.25,
        }
    )


def stream_rows(length: int = 20) -> list[dict[str, float]]:
    """Simulate a live stream with nominal readings followed by a fault."""
    nominal = make_nominal_dataframe(seed=99, length=length)
    faulty = nominal.copy()
    faulty.loc[faulty.index >= length // 2, "motor_right_rpm"] = np.random.default_rng(123).normal(
        0.0,
        1.0,
        length - (length // 2),
    )
    return faulty.to_dict(orient="records")


def main() -> None:
    model = sfdd.SFDDTrainer(theta=0.2).fit_from_dataframes(
        [make_nominal_dataframe(seed=index) for index in range(4)]
    )
    monitor = sfdd.SFDDMonitor(model, window_size=8, step_size=1)

    for index, reading in enumerate(stream_rows(), start=1):
        results = monitor.update(reading)
        for result in results:
            if result.fault_detected:
                print(f"Fault detected at reading {index}: {result.violated_pairs}")
                return

    print("No fault detected in the simulated stream.")


if __name__ == "__main__":
    main()
