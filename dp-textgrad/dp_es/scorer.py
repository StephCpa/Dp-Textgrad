from __future__ import annotations

from dataclasses import dataclass, field
from math import log, sqrt
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union
import random
import hashlib

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
    adaptive_clipping: bool = True  # NEW: Enable adaptive clipping
    adaptive_clipping_quantile: float = 0.95  # NEW: Target quantile for adaptive clipping
    enable_score_cache: bool = True  # NEW: Enable score caching

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
        if self.adaptive_clipping_quantile <= 0 or self.adaptive_clipping_quantile >= 1:
            raise ValueError("adaptive_clipping_quantile must be in (0, 1).")

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
    """Differential privacy aware wrapper for candidate evaluation.

    Optimizations:
    - Adaptive clipping: Dynamically adjusts clipping value based on score distribution
    - Score caching: Avoids re-evaluating identical candidates
    """

    def __init__(
        self,
        config: DPScorerConfig,
        *,
        feedback_sanitiser: Optional[FeedbackSanitiser] = None,
    ):
        self.config = config
        self.feedback_sanitiser = feedback_sanitiser or FeedbackSanitiser()

        # NEW: Score history for adaptive clipping
        self._score_history: List[float] = []

        # NEW: Score cache {candidate_hash -> (raw_score, feedback)}
        self._score_cache: dict[str, Tuple[float, Any]] = {}

    def _hash_candidate(self, candidate: Candidate) -> str:
        """Generate a hash for candidate deduplication."""
        content = candidate.variable.get_value()
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _compute_adaptive_clipping_value(self, raw_scores: List[float]) -> float:
        """Compute adaptive clipping value based on score distribution.

        Uses quantile-based approach to reduce information loss while maintaining DP.
        """
        if not self.config.adaptive_clipping or len(raw_scores) == 0:
            return self.config.clipping_value

        # Add to history for future adaptations
        self._score_history.extend(raw_scores)

        # Keep only recent history (window size = 100)
        if len(self._score_history) > 100:
            self._score_history = self._score_history[-100:]

        # Compute quantile-based clipping
        sorted_abs_scores = sorted(abs(s) for s in self._score_history)
        quantile_idx = int(len(sorted_abs_scores) * self.config.adaptive_clipping_quantile)

        if quantile_idx < len(sorted_abs_scores):
            adaptive_clip = sorted_abs_scores[quantile_idx]
            # Ensure we don't exceed original clipping value (for safety)
            return min(adaptive_clip, self.config.clipping_value)

        return self.config.clipping_value

    def evaluate(
        self,
        candidates: Sequence[Candidate],
        evaluation_fn: Callable[[Candidate], Union[float, Tuple[float, Any]]],
        rng: Optional[random.Random] = None,
        *,
        description: str = "",
    ) -> DPScores:
        """Evaluate and privatise candidate scores.

        Optimized version with:
        - Score caching to avoid redundant evaluations
        - Adaptive clipping based on score distribution

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

        # Step 1: Collect raw scores (with caching)
        raw_scores: List[float] = []
        feedbacks: List[Optional[Any]] = []

        for candidate in candidates:
            # Check cache first
            if self.config.enable_score_cache:
                candidate_hash = self._hash_candidate(candidate)
                if candidate_hash in self._score_cache:
                    raw_score, feedback = self._score_cache[candidate_hash]
                    raw_scores.append(raw_score)
                    feedbacks.append(feedback)
                    continue

            # Evaluate if not cached
            result = evaluation_fn(candidate)
            feedback: Optional[Any] = None
            if isinstance(result, tuple):
                if len(result) != 2:
                    raise ValueError("evaluation_fn must return (score, feedback) when returning tuples.")
                raw_score, feedback = result
            else:
                raw_score = result
            raw_score = float(raw_score)

            # Cache the result
            if self.config.enable_score_cache:
                self._score_cache[candidate_hash] = (raw_score, feedback)

            raw_scores.append(raw_score)
            feedbacks.append(feedback)

        # Step 2: Compute adaptive clipping value
        clipping_value = self._compute_adaptive_clipping_value(raw_scores)
        noise_std = clipping_value * self.config.noise_multiplier

        # Step 3: Apply clipping and noise
        for idx, (candidate, raw_score, feedback) in enumerate(zip(candidates, raw_scores, feedbacks)):
            clipped_score = max(-clipping_value, min(clipping_value, raw_score))
            noise = rng.gauss(0.0, noise_std) if noise_std > 0 else 0.0
            dp_score = clipped_score + noise

            metadata_update = {
                "dp_last_clipped_score": clipped_score,
                "dp_noise": noise,
                "dp_score": dp_score,
                "dp_adaptive_clipping_value": clipping_value,  # NEW: Track adaptive clipping
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

    def clear_cache(self) -> None:
        """Clear the score cache. Useful when starting a new optimization run."""
        self._score_cache.clear()
        self._score_history.clear()
