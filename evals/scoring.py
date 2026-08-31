"""Deterministic scorers.

No model judges a result here. A judge costs money, varies between runs, and
would put the thing under test on both sides of the measurement.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any


def normalise(text: Any) -> str:
    lowered = str(text).lower()
    stripped = "".join(ch for ch in lowered if ch not in string.punctuation)
    without_articles = re.sub(r"\b(a|an|the)\b", " ", stripped)
    return " ".join(without_articles.split())


def exact_match(prediction: Any, answer: Any) -> bool:
    return normalise(prediction) == normalise(answer)


def numeric_match(prediction: Any, answer: Any, tolerance: float = 0.0) -> bool:
    def as_number(value: Any) -> float | None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        found = re.search(r"-?\d+(?:\.\d+)?", str(value))
        return float(found.group()) if found else None

    left, right = as_number(prediction), as_number(answer)
    if left is None or right is None:
        return False
    return abs(left - right) <= tolerance


def f1(prediction: Any, answer: Any) -> float:
    """Token overlap, the usual measure for answers with acceptable variants."""
    predicted = normalise(prediction).split()
    expected = normalise(answer).split()
    if not predicted or not expected:
        return float(predicted == expected)
    shared = Counter(predicted) & Counter(expected)
    overlap = sum(shared.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def best_f1(prediction: Any, answers: list) -> float:
    """Several answers can be right; credit the closest one."""
    return max((f1(prediction, answer) for answer in answers), default=0.0)
