"""Custom exception classes for the standalone SFDD library."""


class SFDDError(Exception):
    """Base exception for SFDD library errors."""


class SegmentationError(SFDDError):
    """Raised when window segmentation fails."""


class DetectionError(SFDDError):
    """Raised when fault detection encounters an error."""


class DiagnosisError(SFDDError):
    """Raised when the diagnosis procedure fails."""


class EvaluationError(SFDDError):
    """Raised when evaluation or metric computation fails."""
