# corrfdd — Correlation-Based Fault Detection Baseline

`corrfdd` is a small Python library for correlation-based fault detection on
robotic sensor streams. It is inspired by the correlation ideas used in
Khalastchi and Kalech (2018), but it is not a faithful implementation of the
published SFDD algorithm from that paper.

What this package actually does:
- learns global pairwise Pearson correlations from nominal data
- checks whether those correlations break inside a test window
- reports the signal pairs whose correlations deviated beyond a threshold
- provides a standalone minimal-hitting-set helper, plus windowing, streaming,
  and optional ROS bag extraction utilities

What it does not implement from the paper:
- the basic SFDD two-half sliding-window heuristic
- explicit pattern recognition such as `stuck`, `drift`, or `fluctuating`
- the extended SFDD offline pattern co-occurrence memory
- the paper's structural dependency model and full diagnosis pipeline

## Installation

```bash
pip install -e .
```

Optional ROS bag support:

```bash
pip install -e ".[ros]"
```

Or install runtime dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import pandas as pd

from corrfdd import CorrelationDetector, CorrelationTrainer

nominal = pd.DataFrame(
    {
        "motor_left_rpm": [100.0, 101.5, 103.0, 104.0],
        "motor_right_rpm": [100.5, 102.0, 103.5, 104.5],
        "imu_pitch": [0.10, 0.12, 0.15, 0.17],
    }
)

model = CorrelationTrainer(theta=0.2).fit_from_dataframes([nominal])

test_window = nominal.copy()
test_window["motor_right_rpm"] = [100.5, 75.0, 130.0, 60.0]

result = CorrelationDetector().predict_dataframe(test_window, model)
print(result.fault_detected)
print(result.violated_pairs)
```

The runnable examples live in [examples/quickstart.py](examples/quickstart.py),
[examples/custom_robot.py](examples/custom_robot.py),
[examples/streaming.py](examples/streaming.py), and
[examples/ros_bag.py](examples/ros_bag.py).

## API Overview

- `corrfdd.detector`: `CorrelationTrainer`, `CorrelationDetector`, `CorrelationModel`, `DetectionResult`
- `corrfdd.monitor`: `CorrelationMonitor` for rolling online detection
- `corrfdd.ros_adapter`: optional ROS bag extraction via `BagExtractor` and `TopicMapping`
- `corrfdd.diagnosis`: `minimal_hitting_sets` helper
- `corrfdd.metrics`: evaluation helpers such as `ConfusionMatrix` and `classification_report`
- `corrfdd.segmentation`: `WindowRecord`, `WindowSegmenter`, `SegmenterConfig`, and action helpers

Backwards-compatible `SFDD*` aliases are still exported for older code, but the
preferred public names are the `Correlation*` ones.

## How It Works

Training:
concatenate nominal windows and retain signal pairs with sufficiently strong
global Pearson correlation.

Detection:
compute Pearson correlations on a test window and flag any retained pair whose
correlation delta exceeds `theta`.

Diagnosis helper:
if you already have conflicts, `minimal_hitting_sets` can enumerate minimal
candidate explanations. This package does not build those conflicts from a
structural system model for you.

## Running Tests

```bash
pytest tests/
```

## References

- Khalastchi, E. and Kalech, M. "A sensor-based approach for fault detection and diagnosis for robotic systems." *Autonomous Robots* 42(6), 2018.
- Reiter, R. "A theory of diagnosis from first principles." *Artificial Intelligence* 32(1), 1987.
