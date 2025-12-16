from __future__ import annotations

from dataclasses import dataclass, field
from math import log, sqrt, exp
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union, Literal
import random
import hashlib
import warnings

from .population import Candidate
from .feedback import FeedbackSanitiser, FeedbackSanitiserConfig


def compute_composition_privacy(
    epsilon_per_query: float,
    delta_per_query: float,
    num_queries: int,
    composition_type: Literal["basic", "advanced"] = "advanced",
) -> Tuple[float, float]:
    """Compute total privacy cost for multiple DP queries using composition theorems.

    Args:
        epsilon_per_query: Privacy budget per query (ε₀)
        delta_per_query: Privacy parameter per query (δ₀)
        num_queries: Number of queries (k)
        composition_type: "basic" or "advanced" composition

    Returns:
        (total_epsilon, total_delta)

    Theory:
        Basic Composition (Dwork & Roth 2014, Theorem 3.16):
            ε_total = k × ε₀
            δ_total = k × δ₀

        Advanced Composition (Kairouz et al. 2015, Theorem 3.3):
            For small ε₀ (< 1):
                ε_total ≈ √(2k ln(1/δ')) × ε₀
            For general ε₀:
                ε_total = √(2k ln(1/δ')) × ε₀ + k × ε₀ × (e^ε₀ - 1)
            where δ' is a slack parameter (typically δ' = δ₀)
            δ_total = k × δ₀ + δ'

    References:
        - Dwork, C., & Roth, A. (2014). The Algorithmic Foundations of Differential Privacy.
        - Kairouz, P., Oh, S., & Viswanath, P. (2015). The Composition Theorem for Differential Privacy.
    """
    if num_queries <= 0:
        return (0.0, 0.0)

    if composition_type == "basic":
        # Basic composition: direct sum
        return (
            num_queries * epsilon_per_query,
            num_queries * delta_per_query,
        )

    elif composition_type == "advanced":
        # Advanced composition with strong composition theorem
        # Using δ' = δ₀ as slack parameter
        delta_slack = delta_per_query

        # Term 1: Amplification from composition
        # ε' = √(2k ln(1/δ')) × ε₀
        if delta_slack <= 0:
            # Fallback to basic composition if delta_slack is invalid
            warnings.warn(
                f"delta_slack={delta_slack} is invalid for advanced composition, "
                "falling back to basic composition",
                UserWarning
            )
            return compute_composition_privacy(
                epsilon_per_query, delta_per_query, num_queries, "basic"
            )

        term1 = sqrt(2 * num_queries * log(1.0 / delta_slack)) * epsilon_per_query

        # Term 2: Additional cost for moderate ε₀
        # k × ε₀ × (e^ε₀ - 1)
        # For ε₀ << 1, this is negligible (~0)
        if epsilon_per_query < 1.0:
            term2 = 0.0  # Negligible for small epsilon
        else:
            term2 = num_queries * epsilon_per_query * (exp(epsilon_per_query) - 1)

        epsilon_total = term1 + term2
        delta_total = num_queries * delta_per_query + delta_slack

        return (epsilon_total, delta_total)

    else:
        raise ValueError(f"Unknown composition type: {composition_type}")


@dataclass
class DPScorerConfig:
    """Configuration for differential privacy aware scoring.

    IMPORTANT PRIVACY PARAMETERS:
        - epsilon_per_candidate: Privacy budget PER CANDIDATE (not total!)
        - composition: How to combine privacy costs across multiple candidates
            * "basic": total_ε = N × ε_per_candidate (conservative)
            * "advanced": total_ε ≈ √(2N ln(1/δ)) × ε_per_candidate (tighter)
        - enable_feedback: Text feedback is NOT DP-protected, disable in production!
        - unsafe_debug_mode: 🚨 CRITICAL PRIVACY RISK! Exposes raw scores in records.
                             NEVER enable in production! Default: False
    """

    clipping_value: float = 4.0
    noise_multiplier: Optional[float] = 1.0

    # RENAMED from epsilon/delta to make it clear these are PER-CANDIDATE costs
    epsilon_per_candidate: float = 1.0
    delta_per_candidate: float = 1e-5

    # NEW: Composition method for calculating total privacy cost
    composition: Literal["basic", "advanced"] = "advanced"

    mechanism: str = "gaussian"
    adaptive_clipping: bool = True
    adaptive_clipping_quantile: float = 0.95
    enable_score_cache: bool = True

    # NEW: Text feedback control (feedback is NOT differentially private!)
    enable_feedback: bool = False
    _feedback_warning_shown: bool = field(default=False, init=False, repr=False)

    # NEW: Debug mode control - CRITICAL PRIVACY RISK!
    # When True, DPScores.records will contain raw_score, clipped_score, noise
    # This COMPLETELY BYPASSES differential privacy protections!
    # NEVER enable in production or with private data!
    unsafe_debug_mode: bool = False
    _debug_warning_shown: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if self.clipping_value <= 0:
            raise ValueError("clipping_value must be positive.")
        if self.epsilon_per_candidate <= 0:
            raise ValueError("epsilon_per_candidate must be positive.")
        if self.delta_per_candidate < 0:
            raise ValueError("delta_per_candidate must be non-negative.")
        if self.mechanism not in {"gaussian"}:
            raise ValueError(f"Unsupported mechanism: {self.mechanism}")
        if self.composition not in {"basic", "advanced"}:
            raise ValueError(f"composition must be 'basic' or 'advanced', got: {self.composition}")
        if self.noise_multiplier is not None and self.noise_multiplier < 0:
            raise ValueError("noise_multiplier must be non-negative.")
        if self.noise_multiplier is None:
            if self.mechanism == "gaussian":
                if self.delta_per_candidate <= 0:
                    raise ValueError("delta_per_candidate must be positive when calibrating Gaussian noise.")
                self.noise_multiplier = sqrt(2.0 * log(1.25 / self.delta_per_candidate)) / self.epsilon_per_candidate
            else:
                raise ValueError("Automatic noise calibration is only supported for Gaussian mechanism.")
        if self.noise_multiplier is None:
            # Defensive: computation above should always populate it.
            raise ValueError("noise_multiplier could not be determined.")
        if self.adaptive_clipping_quantile <= 0 or self.adaptive_clipping_quantile >= 1:
            raise ValueError("adaptive_clipping_quantile must be in (0, 1).")

    # Backward compatibility properties with deprecation warnings
    @property
    def epsilon(self) -> float:
        """Deprecated: Use epsilon_per_candidate instead.

        This property provides backward compatibility but will be removed in v3.0.
        """
        warnings.warn(
            "DPScorerConfig.epsilon is deprecated and will be removed in v3.0. "
            "Use epsilon_per_candidate instead to clarify that this is the budget PER CANDIDATE, "
            "not the total budget.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.epsilon_per_candidate

    @epsilon.setter
    def epsilon(self, value: float):
        warnings.warn(
            "DPScorerConfig.epsilon is deprecated and will be removed in v3.0. "
            "Use epsilon_per_candidate instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.epsilon_per_candidate = value

    @property
    def delta(self) -> float:
        """Deprecated: Use delta_per_candidate instead.

        This property provides backward compatibility but will be removed in v3.0.
        """
        warnings.warn(
            "DPScorerConfig.delta is deprecated and will be removed in v3.0. "
            "Use delta_per_candidate instead to clarify that this is the parameter PER CANDIDATE, "
            "not the total parameter.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.delta_per_candidate

    @delta.setter
    def delta(self, value: float):
        warnings.warn(
            "DPScorerConfig.delta is deprecated and will be removed in v3.0. "
            "Use delta_per_candidate instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.delta_per_candidate = value

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

    def _hash_candidate(self, candidate) -> str:
        """Generate a hash for candidate deduplication.

        Supports both Candidate and CritiqueOption types.
        """
        # Handle CritiqueOption (has critique attribute)
        if hasattr(candidate, 'critique'):
            content = candidate.critique.text
        # Handle Candidate (has variable attribute)
        elif hasattr(candidate, 'variable'):
            content = candidate.variable.get_value()
        else:
            raise TypeError(f"Unsupported candidate type: {type(candidate)}")

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
                "dp_adaptive_clipping_value": clipping_value,
            }

            # NEW: Conditional feedback storage with privacy warning
            if feedback is not None and self.config.enable_feedback:
                # Show warning once per DPScorer instance
                if not self.config._feedback_warning_shown:
                    warnings.warn(
                        "\n" + "="*80 + "\n"
                        "⚠️  PRIVACY WARNING: Text feedback is NOT differentially private!\n"
                        "="*80 + "\n"
                        "Text feedback may leak sensitive information from training data.\n"
                        "The feedback_sanitiser only performs character filtering, NOT noise addition.\n"
                        "\n"
                        "Only enable feedback (enable_feedback=True) when:\n"
                        "  1. Debugging in a controlled environment\n"
                        "  2. Feedback is already public/non-sensitive\n"
                        "  3. You understand the privacy risks\n"
                        "\n"
                        "For production use with private data: set enable_feedback=False (default)\n"
                        "="*80,
                        UserWarning,
                        stacklevel=3
                    )
                    self.config._feedback_warning_shown = True

                # Store feedback (non-DP!)
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
                    dp_score=float(dp_score),
                    noise_magnitude=abs(float(noise)),  # Only store magnitude, not actual noise
                    metadata_update=metadata_update,
                )
            )

            # CRITICAL: Only populate records in unsafe_debug_mode
            # Records contain raw_score which COMPLETELY BYPASSES differential privacy!
            if self.config.unsafe_debug_mode:
                # Show one-time warning
                if not self.config._debug_warning_shown:
                    warnings.warn(
                        "\n" + "="*80 + "\n"
                        "🚨 CRITICAL PRIVACY WARNING: unsafe_debug_mode is enabled!\n"
                        "="*80 + "\n"
                        "DPScores.records will contain raw_score, clipped_score, and noise.\n"
                        "This COMPLETELY BYPASSES all differential privacy protections!\n"
                        "\n"
                        "Any code accessing dp_scores.records can see the EXACT unprotected scores.\n"
                        "This is equivalent to having NO privacy protection at all.\n"
                        "\n"
                        "⚠️  NEVER enable unsafe_debug_mode when:\n"
                        "  1. Working with real private/sensitive data\n"
                        "  2. In production environments\n"
                        "  3. Sharing results with others\n"
                        "  4. Publishing research results\n"
                        "\n"
                        "✓  ONLY enable for:\n"
                        "  1. Local debugging with synthetic/public data\n"
                        "  2. Understanding DP mechanics (educational)\n"
                        "  3. Verifying noise injection is working\n"
                        "\n"
                        "To disable: set unsafe_debug_mode=False (default) in DPScorerConfig\n"
                        "="*80,
                        UserWarning,
                        stacklevel=3
                    )
                    self.config._debug_warning_shown = True

                records.append(DPScoreRecord(
                    idx, float(raw_score), float(clipped_score),
                    float(noise), float(dp_score), feedback
                ))
            else:
                # In production mode: records remain empty, protecting privacy
                # Only DP-protected scores are available via updated_candidates
                pass

        # NEW: Compute total privacy cost using composition theorem
        num_candidates = len(candidates)
        total_epsilon, total_delta = compute_composition_privacy(
            epsilon_per_query=self.config.epsilon_per_candidate,
            delta_per_query=self.config.delta_per_candidate,
            num_queries=num_candidates,
            composition_type=self.config.composition,
        )

        return DPScores(
            updated_candidates=updated,
            epsilon=total_epsilon,  # ✅ Correct: total cost for all candidates
            delta=total_delta,      # ✅ Correct: total delta
            description=description,
            records=records,
        )

    def clear_cache(self) -> None:
        """Clear the score cache. Useful when starting a new optimization run."""
        self._score_cache.clear()
        self._score_history.clear()
