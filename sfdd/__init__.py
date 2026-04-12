"""Public API for the standalone SFDD library."""

from sfdd.detector import (
    TURTLEBOT3_SIGNALS,
    DetectionResult,
    SFDDDetector,
    SFDDModel,
    SFDDTrainer,
)
from sfdd.diagnosis import minimal_hitting_sets
from sfdd.exceptions import (
    DetectionError,
    DiagnosisError,
    EvaluationError,
    SegmentationError,
    SFDDError,
)
from sfdd.metrics import ConfusionMatrix, EvaluationReport, FaultTypeMetrics, classification_report
from sfdd.monitor import SFDDMonitor
from sfdd.segmentation import (
    ACTION_IDLE,
    ACTION_MOVE_BACKWARD,
    ACTION_MOVE_FORWARD,
    ACTION_TURN,
    SegmenterConfig,
    WindowRecord,
    WindowSegmenter,
    classify_action,
)

__all__ = [
    "ACTION_IDLE",
    "ACTION_MOVE_BACKWARD",
    "ACTION_MOVE_FORWARD",
    "ACTION_TURN",
    "ConfusionMatrix",
    "DetectionError",
    "DetectionResult",
    "DiagnosisError",
    "EvaluationError",
    "EvaluationReport",
    "FaultTypeMetrics",
    "SFDDError",
    "SFDDDetector",
    "SFDDModel",
    "SFDDMonitor",
    "SFDDTrainer",
    "SegmenterConfig",
    "SegmentationError",
    "TURTLEBOT3_SIGNALS",
    "WindowRecord",
    "WindowSegmenter",
    "classification_report",
    "classify_action",
    "minimal_hitting_sets",
]
