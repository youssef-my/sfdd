"""Streaming monitor built on top of the correlation detector."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import pandas as pd

from corrfdd.detector import CorrelationDetector, CorrelationModel, DetectionResult
from corrfdd.segmentation import WindowRecord


class CorrelationMonitor:
    """Online correlation monitor for live sensor streams."""

    def __init__(
        self,
        model: CorrelationModel,
        window_size: int = 8,
        step_size: int = 1,
        detector: CorrelationDetector | None = None,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        if step_size < 1:
            raise ValueError("step_size must be at least 1")

        self.model = model
        self.window_size = window_size
        self.step_size = step_size
        self.detector = detector or CorrelationDetector()
        self._buffer: deque[dict[str, float]] = deque(maxlen=window_size)
        self._readings_since_detection = 0

    def update(self, reading: dict[str, float]) -> list[DetectionResult]:
        """Push a new sensor reading and return any new detection results."""
        self._buffer.append(dict(reading))
        self._readings_since_detection += 1

        if len(self._buffer) < self.window_size:
            return []
        if self._readings_since_detection < self.step_size:
            return []

        self._readings_since_detection = 0
        window = WindowRecord(
            action="stream",
            start_time=0.0,
            end_time=0.0,
            data=pd.DataFrame(list(self._buffer)),
            source_bag=Path("stream"),
        )
        return [self.detector.predict(window, self.model)]

    def reset(self) -> None:
        """Clear the internal buffer."""
        self._buffer.clear()
        self._readings_since_detection = 0

    @property
    def buffer_size(self) -> int:
        """Current number of readings in the buffer."""
        return len(self._buffer)


# Backwards-compatible alias for earlier SFDD-branded releases.
SFDDMonitor = CorrelationMonitor
