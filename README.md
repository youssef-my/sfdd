# sfdd — Sensor-based Fault Detection and Diagnosis

`sfdd` is a standalone Python library for the SFDD baseline, a correlation-based
fault detection and diagnosis method for robotic systems based on the approach
described by Khalastchi and Kalech (2018). It learns expected pairwise sensor
correlations from nominal operation and flags windows whose observed
correlations deviate beyond a configurable threshold.

## Installation

Install the package in editable mode:

```bash
pip install -e .
```

Or install the runtime dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
import pandas as pd
from pathlib import Path

from sfdd import SFDDDetector, SFDDTrainer, WindowRecord


def make_window(seed: int, faulty: bool = False) -> WindowRecord:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=40, freq="250ms")
    base = np.linspace(0, 1, len(timestamps)) + rng.normal(0, 0.02, len(timestamps))

    odom_linear_x = base * 0.6
    if faulty:
        odom_linear_x = rng.normal(0.0, 0.6, len(timestamps))

    data = pd.DataFrame(
        {
            "cmd_vel_linear_x": base * 0.6,
            "odom_x": base * 1.3,
            "odom_linear_x": odom_linear_x,
            "sonar_robot": 2.5 - base * 0.4,
            "laser_front": 3.0 - base * 0.6,
        },
        index=timestamps,
    )

    return WindowRecord(
        action="move_forward",
        start_time=timestamps[0].timestamp(),
        end_time=timestamps[-1].timestamp(),
        data=data,
        source_bag=Path(f"synthetic_{seed}.csv"),
    )


nominal_windows = [make_window(seed) for seed in range(5)]
model = SFDDTrainer(theta=0.25).fit(nominal_windows)

result = SFDDDetector().predict(make_window(99, faulty=True), model)
print(result.fault_detected)
print(result.violated_pairs)
```

The repository also includes a runnable example at `examples/quickstart.py`.

## API Overview

- `sfdd.detector`: `SFDDTrainer`, `SFDDDetector`, `SFDDModel`, and `DetectionResult`
- `sfdd.diagnosis`: minimal hitting set diagnosis via `minimal_hitting_sets`
- `sfdd.metrics`: `ConfusionMatrix`, `EvaluationReport`, and `classification_report`
- `sfdd.segmentation`: `WindowRecord`, `WindowSegmenter`, `SegmenterConfig`, and action helpers

## How It Works

Training:
learns global pairwise Pearson correlations from nominal sensor windows.

Detection:
computes correlations for a test window and flags signal pairs whose deviation
from the reference exceeds the per-pair threshold `theta`.

Diagnosis:
maps violated signal-pair conflicts to candidate components and solves for
minimal hitting sets following Reiter's diagnosis formulation.

## Running Tests

```bash
pytest tests/
```

## References

- Khalastchi, E. and Kalech, M. "A sensor-based approach for fault detection and diagnosis for robotic systems." *Autonomous Robots* 42(6), 2018.
- Reiter, R. "A theory of diagnosis from first principles." *Artificial Intelligence* 32(1), 1987.
