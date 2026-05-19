"""Public API for the corrfdd package."""

from corrfdd.detector import (
    CorrelationDetector,
    CorrelationModel,
    CorrelationTrainer,
    TURTLEBOT3_SIGNALS,
    DetectionResult,
    SFDDDetector,
    SFDDModel,
    SFDDTrainer,
)
from corrfdd.diagnosis import minimal_hitting_sets
from corrfdd.exceptions import (
    CorrFDDError,
    DetectionError,
    DiagnosisError,
    EvaluationError,
    SegmentationError,
    SFDDError,
)
from corrfdd.metrics import ConfusionMatrix, EvaluationReport, FaultTypeMetrics, classification_report
from corrfdd.monitor import CorrelationMonitor, SFDDMonitor
from corrfdd.segmentation import (
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
    "CorrFDDError",
    "CorrelationDetector",
    "CorrelationModel",
    "CorrelationMonitor",
    "CorrelationTrainer",
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
