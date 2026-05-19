"""Tests for evaluation metrics."""

from __future__ import annotations

import pytest

from corrfdd.exceptions import EvaluationError
from corrfdd.metrics import ConfusionMatrix, classification_report


class TestConfusionMatrix:
    def test_perfect_classification(self) -> None:
        confusion = ConfusionMatrix(tp=10, fp=0, tn=10, fn=0)
        assert confusion.precision == 1.0
        assert confusion.recall == 1.0
        assert confusion.f1 == 1.0
        assert confusion.accuracy == 1.0

    def test_no_predictions(self) -> None:
        confusion = ConfusionMatrix(tp=0, fp=0, tn=10, fn=5)
        assert confusion.precision == 0.0
        assert confusion.recall == 0.0
        assert confusion.f1 == 0.0

    def test_all_false_positives(self) -> None:
        confusion = ConfusionMatrix(tp=0, fp=10, tn=0, fn=0)
        assert confusion.precision == 0.0

    def test_partial_results(self) -> None:
        confusion = ConfusionMatrix(tp=8, fp=2, tn=7, fn=3)
        assert confusion.precision == pytest.approx(0.8)
        assert confusion.recall == pytest.approx(8 / 11, abs=0.001)


class TestClassificationReport:
    def test_perfect_detection(self) -> None:
        report = classification_report(
            [True, True, False, False], [True, True, False, False], method="test"
        )
        assert report.overall.f1 == 1.0
        assert report.method == "test"
        assert report.total_samples == 4

    def test_all_wrong(self) -> None:
        report = classification_report(
            [True, True, False, False], [False, False, True, True], method="test"
        )
        assert report.overall.precision == 0.0
        assert report.overall.recall == 0.0

    def test_with_fault_types(self) -> None:
        report = classification_report(
            [True, True, True, False, False],
            [True, True, False, False, False],
            fault_types=["odometer", "lidar", "odometer", "", ""],
            method="test",
        )
        assert len(report.per_fault_type) == 2
        fault_types = {fault_type.fault_type for fault_type in report.per_fault_type}
        assert "odometer" in fault_types
        assert "lidar" in fault_types

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(EvaluationError, match="same length"):
            classification_report([True, False], [True])

    def test_mismatched_fault_types_raises(self) -> None:
        with pytest.raises(EvaluationError, match="same length"):
            classification_report([True, False], [True, False], fault_types=["odom"])

    def test_format_table(self) -> None:
        report = classification_report(
            [True, False, True, False], [True, False, False, False], method="test"
        )
        table = report.format_table()
        assert "TEST" in table
        assert "Precision" in table or "P=" in table

    def test_to_dict(self) -> None:
        report = classification_report([True, False], [True, False], method="test")
        as_dict = report.to_dict()
        assert "method" in as_dict
        assert "overall" in as_dict
        assert "macro_f1" in as_dict
