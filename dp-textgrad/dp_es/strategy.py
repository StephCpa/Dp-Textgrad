from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional
import random

from dp_textgrad.variable import Variable
from dp_textgrad.optimizer.optimizer import Optimizer

from .accountant import PrivacyAccountant, PrivacyBudgetExceeded
from .mutation import MutationEngine
from .population import Candidate, PopulationManager
from .scorer import DPScorer, DPScores
from .selector import DPSelector, DPSelectionResult


def _clone_variable(variable: Variable) -> Variable:
    clone = Variable(
        value=variable.get_value(),
        role_description=variable.get_role_description(),
        requires_grad=variable.requires_grad,
    )
    return clone


@dataclass
class DPEvolutionConfig:
    population_size: int = 8
    parents_to_select: int = 4
    max_iterations: int = 10
    stop_on_budget: bool = True
    evaluation_description: str = "dp_es_iteration"
    rng_seed: Optional[int] = None
    evaluation_takes_candidate: bool = False

    def __post_init__(self):
        if self.population_size <= 0:
            raise ValueError("population_size must be positive.")
        if self.parents_to_select <= 0:
            raise ValueError("parents_to_select must be positive.")
        if self.parents_to_select > self.population_size:
            raise ValueError("parents_to_select cannot exceed population_size.")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")


class DPEvolutionStrategy(Optimizer):
    """Differential Privacy aware Evolution Strategy for TextGrad variables."""

    def __init__(
        self,
        parameter: Variable,
        evaluation_fn: Callable[[Variable], float],
        scorer: DPScorer,
        selector: DPSelector,
        mutation_engine: MutationEngine,
        accountant: PrivacyAccountant,
        config: Optional[DPEvolutionConfig] = None,
    ):
        super().__init__([parameter])
        self.parameter = parameter
        self.evaluation_fn = evaluation_fn
        self.dp_scorer = scorer
        self.selector = selector
        self.mutation_engine = mutation_engine
        self.accountant = accountant
        self.config = config or DPEvolutionConfig()
        self.rng = random.Random(self.config.rng_seed)
        self._last_scores: Optional[DPScores] = None
        self.population = self._bootstrap_population(parameter)
        self.best_candidate: Optional[Candidate] = None
        self._iteration = 0
        if hasattr(self.mutation_engine, "bind_accountant"):
            self.mutation_engine.bind_accountant(self.accountant)

    def _bootstrap_population(self, variable: Variable) -> PopulationManager:
        initial_variable = _clone_variable(variable)
        initial_candidate = Candidate(variable=initial_variable, metadata={"candidate_id": "root"})
        candidates: List[Candidate] = [initial_candidate]

        # Expand to desired population size using mutation engine on the root candidate.
        while len(candidates) < self.config.population_size:
            offspring = self.mutation_engine.generate(
                parents=[candidates[-1]],
                iteration=0,
                rng=self.rng,
                feedback=None,
            )
            if not offspring:
                break
            candidates.extend(offspring)

        # If still short, clone root candidates to fill.
        while len(candidates) < self.config.population_size:
            clone = Candidate(variable=_clone_variable(variable), metadata={"candidate_id": f"root-{len(candidates)}"})
            candidates.append(clone)

        return PopulationManager(candidates[: self.config.population_size])

    def _evaluate_population(self) -> DPScores:
        def evaluation_adapter(candidate: Candidate):
            if self.config.evaluation_takes_candidate:
                return self.evaluation_fn(candidate)
            return self.evaluation_fn(candidate.variable)

        scores = self.dp_scorer.evaluate(
            self.population.as_list(),
            evaluation_fn=evaluation_adapter,
            rng=self.rng,
            description=f"{self.config.evaluation_description}:{self._iteration}",
        )
        # Update accountant
        self.accountant.consume(scores.epsilon, scores.delta, description=f"iteration-{self._iteration}")
        self.population.update(scores.updated_candidates)
        self._last_scores = scores
        return scores

    def _select_parents(self) -> List[Candidate]:
        selection = self.selector.select_with_metadata(self.population.as_list(), rng=self.rng)
        # 选择阶段同样需要计入隐私预算；若 epsilon/delta 为 0 则跳过。
        if selection.epsilon > 0 or selection.delta > 0:
            self.accountant.consume(
                selection.epsilon,
                selection.delta,
                description=f"selection-{self._iteration}",
            )
        return selection.selected[: self.config.parents_to_select]

    def _build_next_population(self, parents: Iterable[Candidate]) -> None:
        parent_list = list(parents)
        offspring = self.mutation_engine.generate(
            parents=parent_list,
            iteration=self._iteration,
            rng=self.rng,
            feedback=self._last_scores,
        )
        combined = list(parent_list) + offspring
        if len(combined) < self.config.population_size:
            # Fill the rest with best remaining candidates
            remaining = [
                cand
                for cand in self.population.as_list()
                if cand not in combined
            ]
            remaining.sort(key=lambda c: (c.dp_score is not None, c.dp_score), reverse=True)
            combined.extend(remaining[: self.config.population_size - len(combined)])
        self.population.update(combined[: self.config.population_size])

    def step(self):
        """Run the evolutionary loop until reaching max_iterations or budget exhaustion."""
        for self._iteration in range(1, self.config.max_iterations + 1):
            try:
                self._evaluate_population()
            except PrivacyBudgetExceeded:
                if self.config.stop_on_budget:
                    break
                raise

            parents = self._select_parents()
            if not parents:
                break

            self._build_next_population(parents)
            self._update_best()

        if self.best_candidate is None:
            # Evaluate once to ensure best candidate exists.
            self._update_best()

        if self.best_candidate is not None:
            self.parameter.set_value(self.best_candidate.variable.get_value())

    def _update_best(self) -> None:
        try:
            current_best = self.population.best()
        except ValueError:
            return
        if self.best_candidate is None:
            self.best_candidate = current_best
        else:
            # Compare dp_score first, fallback to raw_score if necessary.
            best_score = self.best_candidate.dp_score if self.best_candidate.dp_score is not None else float("-inf")
            current_score = current_best.dp_score if current_best.dp_score is not None else float("-inf")
            if current_score >= best_score:
                self.best_candidate = current_best
