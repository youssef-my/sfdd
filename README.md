# sfdd — Sensor-based Fault Detection and Diagnosis

`sfdd` is a standalone Python library for correlation-based fault detection and
diagnosis on robotic systems. It was extracted from a TurtleBot3 research
project, but it now works with any robot as long as you can provide nominal
sensor data as pandas DataFrames, windows, ROS bag topics, or live streams.

## Installation

Install the package in editable mode:

```bash
pip install -e .
```

Install the optional ROS bag support when you need bag extraction:

```bash
pip install -e ".[ros]"
```

Or install the runtime dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

The simplest path is to train directly from DataFrames and let `sfdd`
auto-discover the numeric signal columns:

```python
import pandas as pd

from sfdd import SFDDDetector, SFDDTrainer

nominal = pd.DataFrame(
    {
        "motor_left_rpm": [100.0, 101.5, 103.0, 104.0],
        "motor_right_rpm": [100.5, 102.0, 103.5, 104.5],
        "imu_pitch": [0.10, 0.12, 0.15, 0.17],
    }
)

model = SFDDTrainer(theta=0.2).fit_from_dataframes([nominal])

test_window = nominal.copy()
test_window["motor_right_rpm"] = [100.5, 75.0, 130.0, 60.0]

result = SFDDDetector().predict_dataframe(test_window, model)
print(result.fault_detected)
print(result.violated_pairs)
```

The repository includes runnable examples in [examples/quickstart.py](examples/quickstart.py), [examples/custom_robot.py](examples/custom_robot.py), and [examples/streaming.py](examples/streaming.py).

## Custom Robots

Auto-discovery works well for most cases, but you can still pin an explicit
signal schema when you want tighter control:

```python
from sfdd import SFDDTrainer

trainer = SFDDTrainer(
    theta=0.2,
    signal_columns=("motor_left_rpm", "motor_right_rpm", "imu_pitch"),
)
model = trainer.fit_from_dataframes([nominal_dataframe])
```

For backwards compatibility, `sfdd.detector.TURTLEBOT3_SIGNALS` remains
available as a convenience preset.

## Streaming / Live Detection

Use `SFDDMonitor` to keep a rolling window over incoming sensor readings and
run detection online:

```python
from sfdd import SFDDMonitor, SFDDTrainer

model = SFDDTrainer(theta=0.2).fit_from_dataframes([nominal_dataframe])
monitor = SFDDMonitor(model, window_size=8, step_size=1)

for reading in live_sensor_stream:
    for result in monitor.update(reading):
        if result.fault_detected:
            print("FAULT:", result.violated_pairs)
```

## ROS Bag Extraction

ROS bag support is optional and lives in `sfdd.ros_adapter`, so the base
library does not require `rosbags`.

```python
from pathlib import Path

from sfdd.ros_adapter import BagExtractor, turtlebot3_mappings

extractor = BagExtractor(turtlebot3_mappings())
df = extractor.extract(Path("my_recording.bag"))
windows = extractor.extract_to_windows(Path("my_recording.bag"))
```

You can also define your own topic mappings with `TopicMapping`. See
[examples/ros_bag.py](examples/ros_bag.py).

## API Overview

- `sfdd.detector`: `SFDDTrainer`, `SFDDDetector`, `SFDDModel`, `DetectionResult`, `TURTLEBOT3_SIGNALS`
- `sfdd.monitor`: `SFDDMonitor` for streaming / live detection
- `sfdd.ros_adapter`: optional ROS bag extraction via `BagExtractor`, `TopicMapping`, and extractor presets
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
