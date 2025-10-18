from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence
import math
import random

from .population import Candidate


@dataclass
class DPSelectorConfig:
    select_k: int
    epsilon: float = 1.0
    delta: float = 0.0
    sensitivity: float = 1.0
    mechanism: str = "noisy_topk"
    gumbel_scale_override: Optional[float] = None

    def __post_init__(self):
        if self.select_k <= 0:
            raise ValueError("select_k must be positive.")
        if self.sensitivity <= 0:
            raise ValueError("sensitivity must be positive.")
        if self.gumbel_scale_override is not None and self.gumbel_scale_override < 0:
            raise ValueError("gumbel_scale_override must be non-negative when provided.")
        if self.gumbel_scale_override is None:
            if self.epsilon <= 0:
                raise ValueError("epsilon must be positive when automatic noise calibration is used.")
            if self.mechanism not in {"noisy_topk"}:
                raise ValueError(f"Unsupported mechanism: {self.mechanism}")
            if self.delta < 0:
                raise ValueError("delta must be non-negative.")

    @property
    def gumbel_scale(self) -> float:
        if self.gumbel_scale_override is not None:
            return self.gumbel_scale_override
        # For the exponential mechanism implemented via Gumbel sampling the scale equals 2Δu/ε.
        return (2.0 * self.sensitivity) / self.epsilon

    @property
    def privacy_cost(self) -> tuple[float, float]:
        if self.gumbel_scale == 0.0:
            return 0.0, 0.0
        return self.epsilon, self.delta


@dataclass
class DPSelectionRecord:
    index: int
    candidate_id: Optional[str]
    base_score: float
    noise: float
    noisy_score: float


@dataclass
class DPSelectionResult:
    selected: List[Candidate]
    epsilon: float
    delta: float
    mechanism: str
    records: List[DPSelectionRecord] = field(default_factory=list)


class DPSelector:
    """Select parent candidates from a DP scored population."""

    def __init__(self, config: DPSelectorConfig):
        self.config = config

    def _sample_gumbel(self, rng: random.Random) -> float:
        u = rng.random()
        # Clip to avoid log(0)
        u = min(max(u, 1e-12), 1.0 - 1e-12)
        return -math.log(-math.log(u))

    def select(self, candidates: Sequence[Candidate], rng: Optional[random.Random] = None) -> List[Candidate]:
        result = self.select_with_metadata(candidates, rng=rng)
        return result.selected

    def select_with_metadata(
        self,
        candidates: Sequence[Candidate],
        rng: Optional[random.Random] = None,
    ) -> DPSelectionResult:
        if rng is None:
            rng = random

        scored_candidates = []
        records: List[DPSelectionRecord] = []
        gumbel_scale = self.config.gumbel_scale

        if gumbel_scale < 0:
            raise ValueError("Gumbel scale must be non-negative.")

        for candidate in candidates:
            if candidate.dp_score is None:
                raise ValueError("Candidate is missing a dp_score. Ensure DPScorer.evaluate was called first.")
            augmented_score = candidate.dp_score
            noise = 0.0
            if gumbel_scale > 0:
                noise = self._sample_gumbel(rng) * gumbel_scale
                augmented_score += noise
            candidate.metadata["dp_selection_last_noise"] = noise
            candidate.metadata["dp_selection_last_noisy_score"] = augmented_score
            scored_candidates.append((augmented_score, candidate))
            records.append(
                DPSelectionRecord(
                    index=len(records),
                    candidate_id=candidate.metadata.get("candidate_id"),
                    base_score=float(candidate.dp_score),
                    noise=float(noise),
                    noisy_score=float(augmented_score),
                )
            )

        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        k = min(self.config.select_k, len(scored_candidates))
        selected = [candidate for _, candidate in scored_candidates[:k]]

        epsilon, delta = self.config.privacy_cost
        return DPSelectionResult(
            selected=selected,
            epsilon=epsilon,
            delta=delta,
            mechanism=self.config.mechanism,
            records=records,
        )
