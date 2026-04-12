"""Consistency-based diagnosis via minimal hitting sets."""

from __future__ import annotations

import logging
from collections import deque

logger = logging.getLogger(__name__)


def minimal_hitting_sets(conflicts: list[frozenset[str]]) -> list[frozenset[str]]:
    """Compute minimal hitting sets over a collection of conflicts."""
    if not conflicts:
        return []

    conflicts = list({conflict for conflict in conflicts if conflict})
    if not conflicts:
        return []

    conflicts.sort(key=len)

    diagnoses: list[frozenset[str]] = []
    queue: deque[frozenset[str]] = deque([frozenset()])

    while queue:
        candidate = queue.popleft()

        if any(existing <= candidate for existing in diagnoses):
            continue

        unhit = _first_unhit_conflict(candidate, conflicts)
        if unhit is None:
            diagnoses.append(candidate)
            continue

        for component in sorted(unhit):
            new_candidate = candidate | frozenset({component})
            if not any(existing <= new_candidate for existing in diagnoses):
                queue.append(new_candidate)

    return diagnoses


def _first_unhit_conflict(
    candidate: frozenset[str],
    conflicts: list[frozenset[str]],
) -> frozenset[str] | None:
    """Find the first conflict not intersected by the candidate."""
    for conflict in conflicts:
        if not candidate & conflict:
            return conflict
    return None
