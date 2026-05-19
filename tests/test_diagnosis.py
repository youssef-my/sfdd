"""Tests for minimal hitting set diagnosis."""

from __future__ import annotations

from corrfdd.diagnosis import minimal_hitting_sets


class TestMinimalHittingSets:
    def test_empty_conflicts(self) -> None:
        assert minimal_hitting_sets([]) == []

    def test_single_conflict(self) -> None:
        conflicts = [frozenset({"a", "b"})]
        result = minimal_hitting_sets(conflicts)
        assert frozenset({"a"}) in result
        assert frozenset({"b"}) in result
        assert len(result) == 2

    def test_two_overlapping_conflicts(self) -> None:
        conflicts = [
            frozenset({"left_motor", "left_encoder"}),
            frozenset({"lidar", "ultrasonic"}),
        ]
        result = minimal_hitting_sets(conflicts)
        assert len(result) == 4
        for diagnosis in result:
            assert diagnosis & conflicts[0]
            assert diagnosis & conflicts[1]

    def test_subset_conflict_pruning(self) -> None:
        conflicts = [
            frozenset({"a"}),
            frozenset({"a", "b"}),
        ]
        result = minimal_hitting_sets(conflicts)
        assert frozenset({"a"}) in result
        assert all(frozenset({"a"}) <= diagnosis for diagnosis in result)

    def test_single_element_conflict(self) -> None:
        conflicts = [
            frozenset({"a"}),
            frozenset({"b", "c"}),
        ]
        result = minimal_hitting_sets(conflicts)
        for diagnosis in result:
            assert "a" in diagnosis

    def test_paper_example(self) -> None:
        conflicts = [
            frozenset({"left_motor", "left_encoder"}),
            frozenset({"lidar", "ultrasonic"}),
        ]
        result = minimal_hitting_sets(conflicts)

        expected_examples = [
            frozenset({"left_motor", "lidar"}),
            frozenset({"left_encoder", "ultrasonic"}),
        ]
        for expected in expected_examples:
            assert expected in result
