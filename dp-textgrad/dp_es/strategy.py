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
    # NEW: Early stopping parameters
    enable_early_stopping: bool = True
    early_stop_patience: int = 3  # Stop if no improvement for N iterations
    early_stop_threshold: float = 0.001  # Minimum improvement threshold
    # NEW: Elite preservation
    enable_elitism: bool = True
    elite_size: int = 2  # Number of top candidates to preserve

    def __post_init__(self):
        if self.population_size <= 0:
            raise ValueError("population_size must be positive.")
        if self.parents_to_select <= 0:
            raise ValueError("parents_to_select must be positive.")
        if self.parents_to_select > self.population_size:
            raise ValueError("parents_to_select cannot exceed population_size.")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        if self.early_stop_patience <= 0:
            raise ValueError("early_stop_patience must be positive.")
        if self.early_stop_threshold < 0:
            raise ValueError("early_stop_threshold must be non-negative.")
        if self.elite_size < 0 or self.elite_size >= self.population_size:
            raise ValueError("elite_size must be in [0, population_size).")


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
        # NEW: Early stopping tracking
        self._best_score_history: List[float] = []
        self._no_improvement_count: int = 0
        self._converged: bool = False
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
        """Build next generation with optional elitism.

        Optimized to preserve top performers across generations.
        """
        parent_list = list(parents)
        offspring = self.mutation_engine.generate(
            parents=parent_list,
            iteration=self._iteration,
            rng=self.rng,
            feedback=self._last_scores,
        )

        # NEW: Elite preservation
        elites = []
        if self.config.enable_elitism and self.config.elite_size > 0:
            # Preserve top candidates from current population
            current_population = self.population.as_list()
            sorted_by_score = sorted(
                [c for c in current_population if c.dp_score is not None],
                key=lambda c: c.dp_score,
                reverse=True
            )
            elites = sorted_by_score[:self.config.elite_size]

        # Combine: elites + parents + offspring
        combined = elites + list(parent_list) + offspring

        # Remove duplicates (by Variable content hash)
        seen_hashes = set()
        unique_combined = []
        for cand in combined:
            content_hash = hash(cand.variable.get_value())
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_combined.append(cand)

        combined = unique_combined

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

    def _check_convergence(self) -> bool:
        """Check if optimization has converged (early stopping).

        Returns:
            True if converged, False otherwise
        """
        if not self.config.enable_early_stopping:
            return False

        if len(self._best_score_history) < 2:
            return False

        # Check for improvement over recent history
        recent_history = self._best_score_history[-self.config.early_stop_patience:]

        if len(recent_history) < self.config.early_stop_patience:
            return False

        # Calculate improvement
        max_recent = max(recent_history)
        min_recent = min(recent_history)
        improvement = max_recent - min_recent

        # Check if improvement is below threshold
        if improvement < self.config.early_stop_threshold:
            self._no_improvement_count += 1
            if self._no_improvement_count >= self.config.early_stop_patience:
                return True
        else:
            self._no_improvement_count = 0

        return False

    def step(self):
        """Run the evolutionary loop until reaching max_iterations, budget exhaustion, or convergence.

        Optimized with:
        - Early stopping based on convergence detection
        - Elite preservation across generations
        - Better tracking of optimization progress
        """
        for self._iteration in range(1, self.config.max_iterations + 1):
            try:
                self._evaluate_population()
            except PrivacyBudgetExceeded:
                if self.config.stop_on_budget:
                    break
                raise

            self._update_best()

            # Track best score for convergence detection
            if self.best_candidate and self.best_candidate.dp_score is not None:
                self._best_score_history.append(self.best_candidate.dp_score)

            # Check for convergence (early stopping)
            if self._check_convergence():
                self._converged = True
                break

            parents = self._select_parents()
            if not parents:
                break

            self._build_next_population(parents)

        # Final update if needed
        if self.best_candidate is None:
            self._update_best()

        if self.best_candidate is not None:
            self.parameter.set_value(self.best_candidate.variable.get_value())

    def get_optimization_stats(self) -> dict:
        """Get statistics about the optimization run.

        Returns:
            Dictionary with optimization statistics
        """
        stats = {
            "iterations_completed": self._iteration,
            "converged": self._converged,
            "best_score": self.best_candidate.dp_score if self.best_candidate else None,
            "privacy_consumed_epsilon": self.accountant.consumed_epsilon,
            "privacy_consumed_delta": self.accountant.consumed_delta,
            "score_history": self._best_score_history.copy(),
        }

        # Add advanced composition stats if available
        if hasattr(self.accountant, 'get_effective_epsilon'):
            stats["effective_epsilon"] = self.accountant.get_effective_epsilon()

        return stats

    def _update_best(self) -> None:
        try:
            current_best = self.population.best()
        except ValueError:
            return
        if self.best_candidate is None:
            self.best_candidate = current_best
        else:
            # Compare dp_score (only DP-protected scores are stored)
            best_score = self.best_candidate.dp_score if self.best_candidate.dp_score is not None else float("-inf")
            current_score = current_best.dp_score if current_best.dp_score is not None else float("-inf")
            if current_score >= best_score:
                self.best_candidate = current_best
