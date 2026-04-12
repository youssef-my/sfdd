"""Binary evaluation metrics for SFDD fault detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sfdd.exceptions import EvaluationError


@dataclass(frozen=True)
class ConfusionMatrix:
    """Binary confusion matrix counts."""

    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)."""
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)."""
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        """F1 = 2 * precision * recall / (precision + recall)."""
        precision, recall = self.precision, self.recall
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """Accuracy = (TP + TN) / total."""
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0


@dataclass(frozen=True)
class FaultTypeMetrics:
    """Metrics for a single fault type."""

    fault_type: str
    confusion: ConfusionMatrix
    support: int

    @property
    def precision(self) -> float:
        return self.confusion.precision

    @property
    def recall(self) -> float:
        return self.confusion.recall

    @property
    def f1(self) -> float:
        return self.confusion.f1


@dataclass(frozen=True)
class EvaluationReport:
    """Full evaluation report with per-fault-type and aggregate metrics."""

    method: str
    overall: ConfusionMatrix
    per_fault_type: list[FaultTypeMetrics]
    macro_precision: float
    macro_recall: float
    macro_f1: float
    total_samples: int

    def to_dict(self) -> dict:  # type: ignore[type-arg]
        """Convert to a JSON-serializable dict."""
        return {
            "method": self.method,
            "overall": {
                "tp": self.overall.tp,
                "fp": self.overall.fp,
                "tn": self.overall.tn,
                "fn": self.overall.fn,
                "precision": self.overall.precision,
                "recall": self.overall.recall,
                "f1": self.overall.f1,
                "accuracy": self.overall.accuracy,
            },
            "per_fault_type": [
                {
                    "fault_type": fault_type.fault_type,
                    "precision": fault_type.precision,
                    "recall": fault_type.recall,
                    "f1": fault_type.f1,
                    "support": fault_type.support,
                }
                for fault_type in self.per_fault_type
            ],
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "total_samples": self.total_samples,
        }

    def format_table(self) -> str:
        """Format the report as a compact text table."""
        lines: list[str] = [f"=== {self.method.upper()} Evaluation Report ===", ""]
        lines.append(
            f"Overall:  P={self.overall.precision:.3f}  "
            f"R={self.overall.recall:.3f}  "
            f"F1={self.overall.f1:.3f}  "
            f"Acc={self.overall.accuracy:.3f}  "
            f"(n={self.total_samples})"
        )
        lines.append("")

        if self.per_fault_type:
            lines.append(
                f"{'Fault Type':<15} {'Precision':>10} {'Recall':>10} "
                f"{'F1':>10} {'Support':>10}"
            )
            lines.append("-" * 55)
            for fault_type in self.per_fault_type:
                lines.append(
                    f"{fault_type.fault_type:<15} {fault_type.precision:>10.3f} "
                    f"{fault_type.recall:>10.3f} {fault_type.f1:>10.3f} "
                    f"{fault_type.support:>10}"
                )
            lines.append("-" * 55)

        lines.append(
            f"{'Macro avg':<15} {self.macro_precision:>10.3f} "
            f"{self.macro_recall:>10.3f} {self.macro_f1:>10.3f}"
        )
        lines.append("")
        lines.append("Confusion Matrix:")
        lines.append(f"  TP={self.overall.tp}  FP={self.overall.fp}")
        lines.append(f"  FN={self.overall.fn}  TN={self.overall.tn}")
        return "\n".join(lines)


def classification_report(
    y_true: list[bool],
    y_pred: list[bool],
    fault_types: list[str] | None = None,
    method: str = "unknown",
) -> EvaluationReport:
    """Compute classification metrics for fault detection."""
    if len(y_true) != len(y_pred):
        raise EvaluationError(
            f"y_true and y_pred must have same length, got {len(y_true)} and {len(y_pred)}"
        )

    if fault_types is not None and len(fault_types) != len(y_true):
        raise EvaluationError(
            "fault_types must have same length as y_true, "
            f"got {len(fault_types)} and {len(y_true)}"
        )

    tp = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth and pred)
    fp = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if not truth and pred)
    fn = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth and not pred)
    tn = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if not truth and not pred)

    overall = ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn)
    per_fault: list[FaultTypeMetrics] = []

    if fault_types is not None:
        unique_types = sorted(
            {
                fault_type
                for fault_type, truth in zip(fault_types, y_true, strict=True)
                if truth and fault_type
            }
        )

        for fault_type in unique_types:
            ft_tp = sum(
                1
                for truth, pred, current_type in zip(y_true, y_pred, fault_types, strict=True)
                if current_type == fault_type and truth and pred
            )
            ft_fn = sum(
                1
                for truth, pred, current_type in zip(y_true, y_pred, fault_types, strict=True)
                if current_type == fault_type and truth and not pred
            )
            ft_support = sum(
                1
                for current_type, truth in zip(fault_types, y_true, strict=True)
                if current_type == fault_type and truth
            )
            per_fault.append(
                FaultTypeMetrics(
                    fault_type=fault_type,
                    confusion=ConfusionMatrix(tp=ft_tp, fp=fp, tn=tn, fn=ft_fn),
                    support=ft_support,
                )
            )

    if per_fault:
        macro_precision = float(np.mean([fault_type.precision for fault_type in per_fault]))
        macro_recall = float(np.mean([fault_type.recall for fault_type in per_fault]))
        macro_f1 = float(np.mean([fault_type.f1 for fault_type in per_fault]))
    else:
        macro_precision = overall.precision
        macro_recall = overall.recall
        macro_f1 = overall.f1

    return EvaluationReport(
        method=method,
        overall=overall,
        per_fault_type=per_fault,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        total_samples=len(y_true),
    )
