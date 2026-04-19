"""Simple predictor for human-generated pseudo-random binary choices.

Humans often avoid long streaks and switch choices more than true randomness.
This module uses a tiny transition model to exploit that tendency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import List, Sequence, Tuple


@dataclass
class TransitionPredictor:
    """Predict next binary choice from observed transitions."""

    transition_counts: List[List[int]] = field(
        default_factory=lambda: [[1, 1], [1, 1]]
    )  # Laplace smoothing

    @staticmethod
    def _validate_binary(value: int, field_name: str) -> None:
        if value not in (0, 1):
            raise ValueError(f"{field_name} must be 0 or 1")

    def predict_next(self, previous_choice: int) -> int:
        """Predict the next binary choice from transition frequencies.

        Uses observed counts for transitions from `previous_choice`.
        If counts are tied, this deterministically predicts repetition.
        """
        self._validate_binary(previous_choice, "previous_choice")
        zero_count, one_count = self.transition_counts[previous_choice]
        if one_count == zero_count:
            return previous_choice
        return 1 if one_count > zero_count else 0

    def observe(self, previous_choice: int, next_choice: int) -> None:
        """Record an observed transition to update predictor state."""
        self._validate_binary(previous_choice, "previous_choice")
        self._validate_binary(next_choice, "next_choice")
        self.transition_counts[previous_choice][next_choice] += 1


def exploit_human_random(sequence: Sequence[int]) -> Tuple[List[int], float]:
    """Predict each next move and return predictions and accuracy.

    Args:
        sequence: Binary sequence (0/1) representing human choices.

    Returns:
        (predictions, accuracy)
    """
    if len(sequence) < 2:
        return [], 0.0

    predictor = TransitionPredictor()
    predictions: List[int] = []
    correct = 0

    previous = sequence[0]
    for actual in sequence[1:]:
        prediction = predictor.predict_next(previous)
        predictions.append(prediction)
        if prediction == actual:
            correct += 1
        predictor.observe(previous, actual)
        previous = actual

    accuracy = correct / len(predictions)
    return predictions, accuracy


def generate_human_like_sequence(
    length: int, *, switch_bias: float = 0.7, seed: int = 0
) -> List[int]:
    """Generate a pseudo-random sequence with human-like over-switching tendency."""
    if length < 0:
        raise ValueError("length must be non-negative")
    if not 0.0 <= switch_bias <= 1.0:
        raise ValueError("switch_bias must be between 0.0 and 1.0")
    if length == 0:
        return []

    rng = random.Random(seed)
    sequence = [rng.randint(0, 1)]
    for _ in range(length - 1):
        if rng.random() < switch_bias:
            sequence.append(1 - sequence[-1])
        else:
            sequence.append(sequence[-1])
    return sequence


def demo(length: int = 200, switch_bias: float = 0.7, seed: int = 0) -> str:
    """Run a short demonstration and return a summary string."""
    sequence = generate_human_like_sequence(
        length=length, switch_bias=switch_bias, seed=seed
    )
    _, accuracy = exploit_human_random(sequence)
    return (
        f"Predicted {length - 1} steps with {accuracy:.1%} accuracy "
        f"against human-like switch_bias={switch_bias}."
    )


if __name__ == "__main__":
    print(demo())
