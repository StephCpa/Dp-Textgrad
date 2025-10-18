from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional
import random

from dp_textgrad.variable import Variable

from .population import Candidate
from .critique import CritiquePipeline, CritiqueOption
from .accountant import PrivacyAccountant


def _clone_variable(variable: Variable) -> Variable:
    """Create a shallow clone of a Variable for mutation purposes."""
    clone = Variable(
        value=variable.get_value(),
        role_description=variable.get_role_description(),
        requires_grad=variable.requires_grad,
    )
    return clone


@dataclass
class MutationConfig:
    offspring_per_parent: int = 1
    allow_identity_offspring: bool = False
    critique_description: str = "dp_es_mutation"

    def __post_init__(self):
        if self.offspring_per_parent <= 0:
            raise ValueError("offspring_per_parent must be positive.")


class MutationEngine:
    """Generate new candidate variables from parents."""

    def __init__(
        self,
        mutation_fn: Optional[
            Callable[[Candidate, int, random.Random, Optional[Any]], List[Variable]]
        ] = None,
        config: Optional[MutationConfig] = None,
        *,
        critique_pipeline: Optional[CritiquePipeline] = None,
    ):
        self.mutation_fn = mutation_fn or self._default_mutation_fn
        self.config = config or MutationConfig()
        self.critique_pipeline = critique_pipeline
        self._counter = 0
        self._accountant: Optional[PrivacyAccountant] = None

    def _default_mutation_fn(
        self,
        parent: Candidate,
        iteration: int,
        rng: random.Random,
        feedback: Optional[Any] = None,
    ) -> List[Variable]:
        """Fallback mutation: clone the parent variable."""
        clone = _clone_variable(parent.variable)
        if not self.config.allow_identity_offspring:
            # Introduce a minimal perturbation by appending an iteration marker.
            clone.set_value(f"{clone.get_value()} [iter={iteration}]")
        return [clone]

    def bind_accountant(self, accountant: PrivacyAccountant) -> None:
        """Attach a privacy accountant so critique selection can record costs."""
        self._accountant = accountant

    def generate(
        self,
        parents: Iterable[Candidate],
        iteration: int,
        rng: Optional[random.Random] = None,
        feedback: Optional[Any] = None,
    ) -> List[Candidate]:
        """Create offspring candidates from the supplied parents.

        Args:
            parents: Iterable of parent candidates.
            iteration: The current optimisation iteration.
            rng: Optional RNG; defaults to :mod:`random`.
            feedback: Optional auxiliary signal (DP scores, textual hints, etc.).
        """
        if rng is None:
            rng = random

        offspring: List[Candidate] = []
        for parent in parents:
            critique_payload: Optional[CritiqueOption] = None
            if self.critique_pipeline is not None:
                critique_payload, _, _ = self.critique_pipeline.run(
                    parent=parent,
                    iteration=iteration,
                    rng=rng,
                    feedback=feedback,
                    accountant=self._accountant,
                    description=f"{self.config.critique_description}-{iteration}",
                )
                if critique_payload is not None:
                    parent.metadata["dp_selected_critique"] = critique_payload.critique.text
                    parent.metadata["dp_selected_critique_metadata"] = critique_payload.metadata

            variables = self.mutation_fn(parent, iteration, rng, feedback)
            if not variables:
                continue
            if len(variables) < self.config.offspring_per_parent:
                # Repeat to satisfy offspring count if necessary.
                repeats = (self.config.offspring_per_parent + len(variables) - 1) // len(variables)
                variables = (variables * repeats)[: self.config.offspring_per_parent]
            else:
                variables = variables[: self.config.offspring_per_parent]

            for variable in variables:
                candidate_id = f"cand-{self._counter}"
                self._counter += 1
                metadata = dict(parent.metadata)
                metadata.update({"candidate_id": candidate_id, "parent_id": parent.metadata.get("candidate_id")})
                offspring.append(
                    Candidate(
                        variable=variable,
                        parent_id=parent.metadata.get("candidate_id"),
                        metadata=metadata,
                    )
                )
        return offspring
