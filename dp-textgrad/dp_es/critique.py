from __future__ import annotations

"""Differentially private critique generation and selection utilities."""

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union
import random

from .accountant import PrivacyAccountant
from .feedback import FeedbackSanitiser
from .scorer import DPScorer, DPScores
from .selector import DPSelector, DPSelectionResult


@dataclass
class Critique:
    """Container for an individual critique text and associated metadata."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CritiqueOption:
    """Wrapper making critiques compatible with DP scoring/selection helpers."""

    critique: Critique
    parent_id: Optional[str] = None
    raw_score: Optional[float] = None
    dp_score: Optional[float] = None
    noise: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_scores(
        self,
        raw_score: float,
        dp_score: float,
        noise: float,
        *,
        metadata_update: Optional[dict[str, Any]] = None,
    ) -> "CritiqueOption":
        merged_metadata = dict(self.metadata)
        if metadata_update:
            merged_metadata.update(metadata_update)
        return CritiqueOption(
            critique=self.critique,
            parent_id=self.parent_id,
            raw_score=raw_score,
            dp_score=dp_score,
            noise=noise,
            metadata=merged_metadata,
        )


class CritiquePipeline:
    """End-to-end pipeline that generates, scores, and privately selects critiques."""

    def __init__(
        self,
        generator_fn: Callable[[Any, int, random.Random, Optional[Any]], Sequence[Critique]],
        evaluation_fn: Callable[[Any, Critique], Union[float, Tuple[float, Any]]],
        dp_scorer: DPScorer,
        dp_selector: DPSelector,
        *,
        feedback_sanitiser: Optional[FeedbackSanitiser] = None,
    ) -> None:
        self.generator_fn = generator_fn
        self.evaluation_fn = evaluation_fn
        self.dp_scorer = dp_scorer
        self.dp_selector = dp_selector
        self.feedback_sanitiser = feedback_sanitiser or FeedbackSanitiser()

    def _options_from_critiques(
        self,
        parent_identifier: Optional[str],
        critiques: Iterable[Critique],
        *,
        sanitise: bool = True,
    ) -> List[CritiqueOption]:
        options: List[CritiqueOption] = []
        for index, critique in enumerate(critiques):
            text = critique.text
            if sanitise:
                text = self.feedback_sanitiser.sanitise(text)
            metadata = dict(critique.metadata)
            metadata.setdefault("critique_id", f"crit-{parent_identifier}-{index}")
            metadata["critique_text"] = text
            options.append(
                CritiqueOption(
                    critique=Critique(text=text, metadata=metadata),
                    parent_id=parent_identifier,
                    metadata=metadata,
                )
            )
        return options

    def run(
        self,
        *,
        parent: Any,
        iteration: int,
        rng: Optional[random.Random],
        feedback: Optional[Any],
        accountant: Optional[PrivacyAccountant],
        description: str,
    ) -> tuple[Optional[CritiqueOption], Optional[DPScores], Optional[DPSelectionResult]]:
        """Generate critiques and privately pick one for downstream use."""
        if rng is None:
            rng = random

        critiques = list(self.generator_fn(parent, iteration, rng, feedback))
        if not critiques:
            return None, None, None

        parent_id = getattr(parent, "metadata", {}).get("candidate_id") if parent is not None else None
        options = self._options_from_critiques(parent_id, critiques)

        def evaluation_adapter(option: CritiqueOption):
            return self.evaluation_fn(parent, option.critique)

        scores = self.dp_scorer.evaluate(
            options,
            evaluation_fn=evaluation_adapter,
            rng=rng,
            description=f"{description}:critique",
        )
        if accountant is not None:
            accountant.consume(scores.epsilon, scores.delta, description=f"{description}:critique_score")

        selection = self.dp_selector.select_with_metadata(scores.updated_candidates, rng=rng)
        if accountant is not None and (selection.epsilon > 0 or selection.delta > 0):
            accountant.consume(selection.epsilon, selection.delta, description=f"{description}:critique_select")

        selected = selection.selected[0] if selection.selected else None
        return selected, scores, selection
