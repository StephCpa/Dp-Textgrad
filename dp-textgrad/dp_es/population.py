from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, List, Optional, Sequence

from dp_textgrad.variable import Variable


@dataclass
class Candidate:
    """Container tracking a candidate Variable through the DP-ES pipeline."""

    variable: Variable
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
    ) -> "Candidate":
        new_metadata = dict(self.metadata)
        if metadata_update:
            new_metadata.update(metadata_update)
        return replace(self, raw_score=raw_score, dp_score=dp_score, noise=noise, metadata=new_metadata)

    def new_generation(self, variable: Variable, *, metadata: Optional[dict[str, Any]] = None) -> "Candidate":
        """Return a child candidate carrying parent metadata forward."""
        child_metadata = dict(self.metadata)
        if metadata:
            child_metadata.update(metadata)
        return Candidate(variable=variable, parent_id=self.metadata.get("candidate_id"), metadata=child_metadata)


class PopulationManager:
    """Utility for managing a collection of candidates."""

    def __init__(self, candidates: Sequence[Candidate]):
        if not candidates:
            raise ValueError("PopulationManager requires at least one candidate.")
        self._candidates: List[Candidate] = list(candidates)

    @classmethod
    def from_variables(cls, variables: Iterable[Variable]) -> "PopulationManager":
        return cls([Candidate(variable=v, metadata={"candidate_id": f"init-{idx}"}) for idx, v in enumerate(variables)])

    def as_list(self) -> List[Candidate]:
        return list(self._candidates)

    def __iter__(self):
        return iter(self._candidates)

    def best(self, key: str = "dp_score") -> Candidate:
        if not self._candidates:
            raise ValueError("Empty population.")
        return max(
            (c for c in self._candidates if getattr(c, key) is not None),
            key=lambda cand: getattr(cand, key),
        )

    def update(self, new_candidates: Sequence[Candidate]) -> None:
        if not new_candidates:
            raise ValueError("Cannot update with empty candidate list.")
        self._candidates = list(new_candidates)

    def topk(self, k: int, key: str = "dp_score") -> List[Candidate]:
        if k <= 0:
            return []
        sorted_candidates = sorted(
            (c for c in self._candidates if getattr(c, key) is not None),
            key=lambda cand: getattr(cand, key),
            reverse=True,
        )
        return sorted_candidates[:k]

    def size(self) -> int:
        return len(self._candidates)
