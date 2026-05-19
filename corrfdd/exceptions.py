"""Custom exception classes for the corrfdd library."""


class CorrFDDError(Exception):
    """Base exception for corrfdd library errors."""


class SegmentationError(CorrFDDError):
    """Raised when window segmentation fails."""


class DetectionError(CorrFDDError):
    """Raised when fault detection encounters an error."""


class DiagnosisError(CorrFDDError):
    """Raised when the diagnosis procedure fails."""


class EvaluationError(CorrFDDError):
    """Raised when evaluation or metric computation fails."""


# Backwards-compatible alias for earlier SFDD-branded releases.
SFDDError = CorrFDDError
