"""SFDD baseline: Sensor-based Fault Detection and Diagnosis."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from sfdd.exceptions import DetectionError
from sfdd.segmentation import WindowRecord

logger = logging.getLogger(__name__)

DEFAULT_THETA = 0.3
MIN_CORRELATION_THRESHOLD = 0.5
MIN_CORRELATION_SAMPLES = 4

SIGNAL_COLUMNS = (
    "cmd_vel_linear_x",
    "cmd_vel_angular_z",
    "odom_x",
    "odom_y",
    "odom_linear_x",
    "odom_angular_z",
    "sonar_robot",
    "laser_front",
    "laser_frontleft",
    "laser_left",
    "laser_backleft",
    "laser_back",
    "laser_backright",
    "laser_right",
    "laser_frontright",
)


@dataclass(frozen=True)
class DetectionResult:
    """Result of a fault detection check."""

    fault_detected: bool
    violated_pairs: list[tuple[str, str]]
    correlation_deltas: dict[tuple[str, str], float]
    method: str


@dataclass
class SFDDModel:
    """Reference correlations learned from nominal data."""

    reference_correlations: dict[tuple[str, str], float] = field(default_factory=dict)
    theta: dict[tuple[str, str], float] = field(default_factory=dict)
    signal_columns: tuple[str, ...] = SIGNAL_COLUMNS

    def to_dict(self) -> dict:  # type: ignore[type-arg]
        """Serialize to a JSON-compatible dict."""
        return {
            "reference_correlations": {
                f"{key[0]}|{key[1]}": value for key, value in self.reference_correlations.items()
            },
            "theta": {f"{key[0]}|{key[1]}": value for key, value in self.theta.items()},
            "signal_columns": list(self.signal_columns),
        }

    @classmethod
    def from_dict(cls, data: dict) -> SFDDModel:  # type: ignore[type-arg]
        """Deserialize from a JSON-parsed dict."""
        reference_correlations = {
            tuple(key.split("|")): value  # type: ignore[misc]
            for key, value in data.get("reference_correlations", {}).items()
        }
        theta = {
            tuple(key.split("|")): value  # type: ignore[misc]
            for key, value in data.get("theta", {}).items()
        }
        return cls(
            reference_correlations=reference_correlations,  # type: ignore[arg-type]
            theta=theta,  # type: ignore[arg-type]
            signal_columns=tuple(data.get("signal_columns", SIGNAL_COLUMNS)),
        )

    def save(self, path: Path) -> None:
        """Save the model to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load(cls, path: Path) -> SFDDModel:
        """Load the model from a JSON file."""
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def _get_available_signals(df: pd.DataFrame, signal_columns: tuple[str, ...]) -> list[str]:
    """Get signal columns that are present and have data."""
    return [
        column for column in signal_columns if column in df.columns and df[column].notna().sum() > 0
    ]


def _compute_pairwise_correlations(
    df: pd.DataFrame,
    signals: list[str],
) -> dict[tuple[str, str], float]:
    """Compute Pearson correlations for all signal pairs."""
    correlations: dict[tuple[str, str], float] = {}

    for signal_a, signal_b in combinations(signals, 2):
        valid = df[[signal_a, signal_b]].dropna()
        if len(valid) < MIN_CORRELATION_SAMPLES:
            continue

        if valid[signal_a].std() < 1e-10 or valid[signal_b].std() < 1e-10:
            correlations[(signal_a, signal_b)] = 0.0
            continue

        correlation = valid[signal_a].corr(valid[signal_b])
        if np.isnan(correlation):
            correlation = 0.0
        correlations[(signal_a, signal_b)] = float(correlation)

    return correlations


class SFDDTrainer:
    """Train a global SFDD model from nominal windows."""

    def __init__(
        self,
        theta: float = DEFAULT_THETA,
        signal_columns: tuple[str, ...] = SIGNAL_COLUMNS,
    ) -> None:
        self.theta = theta
        self.signal_columns = signal_columns

    def fit(
        self,
        nominal_windows: list[WindowRecord],
        min_correlation_threshold: float = MIN_CORRELATION_THRESHOLD,
    ) -> SFDDModel:
        """Learn reference correlations from nominal training windows."""
        if not nominal_windows:
            raise DetectionError("No nominal windows provided for training")

        all_data = pd.concat([window.data for window in nominal_windows], ignore_index=True)

        signals = _get_available_signals(all_data, self.signal_columns)
        if len(signals) < 2:
            raise DetectionError(f"Need at least 2 signal columns, found {len(signals)}: {signals}")

        reference_correlations = _compute_pairwise_correlations(all_data, signals)
        if not reference_correlations:
            raise DetectionError("No valid correlations could be computed from nominal data")

        filtered_correlations = {
            pair: correlation
            for pair, correlation in reference_correlations.items()
            if abs(correlation) >= min_correlation_threshold
        }

        logger.info(
            "Retained %d/%d global SFDD pairs with |r| >= %.2f",
            len(filtered_correlations),
            len(reference_correlations),
            min_correlation_threshold,
        )

        theta_map = {pair: self.theta for pair in filtered_correlations}
        model = SFDDModel(
            reference_correlations=filtered_correlations,
            theta=theta_map,
            signal_columns=self.signal_columns,
        )

        logger.info(
            "Trained SFDD model: %d retained reference correlations from %d windows",
            len(filtered_correlations),
            len(nominal_windows),
        )
        return model


class SFDDDetector:
    """Detect faults using the SFDD correlation-based method."""

    def predict(self, window: WindowRecord, model: SFDDModel) -> DetectionResult:
        """Check a window for faults using SFDD."""
        signals = _get_available_signals(window.data, model.signal_columns)
        observed = _compute_pairwise_correlations(window.data, signals)

        violated_pairs: list[tuple[str, str]] = []
        deltas: dict[tuple[str, str], float] = {}

        for pair, reference_correlation in model.reference_correlations.items():
            if pair not in observed:
                continue

            delta = abs(observed[pair] - reference_correlation)
            threshold = model.theta.get(pair, DEFAULT_THETA)
            if delta > threshold:
                violated_pairs.append(pair)
            deltas[pair] = delta

        return DetectionResult(
            fault_detected=bool(violated_pairs),
            violated_pairs=violated_pairs,
            correlation_deltas=deltas,
            method="sfdd",
        )
