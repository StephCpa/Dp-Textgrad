from __future__ import annotations

from dataclasses import dataclass, field
from math import log, sqrt
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union
import random

from .population import Candidate
from .feedback import FeedbackSanitiser, FeedbackSanitiserConfig


@dataclass
class DPScorerConfig:
    """Configuration for differential privacy aware scoring."""

    clipping_value: float = 4.0
    noise_multiplier: Optional[float] = 1.0
    epsilon: float = 1.0
    delta: float = 1e-5
    mechanism: str = "gaussian"

    def __post_init__(self):
        if self.clipping_value <= 0:
            raise ValueError("clipping_value must be positive.")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if self.delta < 0:
            raise ValueError("delta must be non-negative.")
        if self.mechanism not in {"gaussian"}:
            raise ValueError(f"Unsupported mechanism: {self.mechanism}")
        if self.noise_multiplier is not None and self.noise_multiplier < 0:
            raise ValueError("noise_multiplier must be non-negative.")
        if self.noise_multiplier is None:
            if self.mechanism == "gaussian":
                if self.delta <= 0:
                    raise ValueError("delta must be positive when calibrating Gaussian noise.")
                self.noise_multiplier = sqrt(2.0 * log(1.25 / self.delta)) / self.epsilon
            else:
                raise ValueError("Automatic noise calibration is only supported for Gaussian mechanism.")
        if self.noise_multiplier is None:
            # Defensive: computation above should always populate it.
            raise ValueError("noise_multiplier could not be determined.")

    @property
    def noise_std(self) -> float:
        return self.clipping_value * self.noise_multiplier


@dataclass
class DPScoreRecord:
    index: int
    raw_score: float
    clipped_score: float
    noise: float
    dp_score: float
    feedback: Optional[Any] = None


@dataclass
class DPScores:
    updated_candidates: List[Candidate]
    epsilon: float
    delta: float
    description: str = ""
    records: List[DPScoreRecord] = field(default_factory=list)


class DPScorer:
    """Differential privacy aware wrapper for candidate evaluation."""

    def __init__(
        self,
        config: DPScorerConfig,
        *,
        feedback_sanitiser: Optional[FeedbackSanitiser] = None,
    ):
        self.config = config
        self.feedback_sanitiser = feedback_sanitiser or FeedbackSanitiser()

    def evaluate(
        self,
        candidates: Sequence[Candidate],
        evaluation_fn: Callable[[Candidate], Union[float, Tuple[float, Any]]],
        rng: Optional[random.Random] = None,
        *,
        description: str = "",
    ) -> DPScores:
        """Evaluate and privatise candidate scores.

        Args:
            candidates: The population under evaluation.
            evaluation_fn: Callable returning a raw score for each candidate.
            rng: Optional random number generator (defaults to global RNG).
            description: Optional textual description of this evaluation call.

        Returns:
            DPScores containing updated candidates (with DP scores) and the privacy cost.
        """
        if rng is None:
            rng = random

        updated: List[Candidate] = []
        records: List[DPScoreRecord] = []
        noise_std = self.config.noise_std

        for idx, candidate in enumerate(candidates):
            result = evaluation_fn(candidate)
            feedback: Optional[Any] = None
            if isinstance(result, tuple):
                if len(result) != 2:
                    raise ValueError("evaluation_fn must return (score, feedback) when returning tuples.")
                raw_score, feedback = result
            else:
                raw_score = result
            raw_score = float(raw_score)
            clipped_score = max(-self.config.clipping_value, min(self.config.clipping_value, raw_score))
            noise = rng.gauss(0.0, noise_std) if noise_std > 0 else 0.0
            dp_score = clipped_score + noise
            metadata_update = {
                "dp_last_clipped_score": clipped_score,
                "dp_noise": noise,
                "dp_score": dp_score,
            }
            if feedback is not None:
                if isinstance(feedback, str):
                    metadata_update["dp_feedback"] = self.feedback_sanitiser.sanitise(feedback)
                elif isinstance(feedback, (list, tuple)):
                    metadata_update["dp_feedback_batch"] = [
                        self.feedback_sanitiser.sanitise(str(item)) for item in feedback
                    ]
                else:
                    metadata_update["dp_feedback"] = feedback
            updated.append(
                candidate.with_scores(
                    raw_score=float(raw_score),
                    dp_score=float(dp_score),
                    noise=float(noise),
                    metadata_update=metadata_update,
                )
            )
            records.append(DPScoreRecord(idx, float(raw_score), float(clipped_score), float(noise), float(dp_score), feedback))

        return DPScores(
            updated_candidates=updated,
            epsilon=self.config.epsilon,
            delta=self.config.delta,
            description=description,
            records=records,
        )
