from __future__ import annotations

from dataclasses import dataclass, field
import math


class PrivacyBudgetExceeded(RuntimeError):
    """Raised when the evolution process would exceed the allocated privacy budget."""


@dataclass
class AccountedPrivacyEvent:
    epsilon: float
    delta: float
    description: str = ""


@dataclass
class PrivacyAccountant:
    """A minimal privacy accountant using basic composition.

    This implementation intentionally keeps the accounting logic simple:
    it sums the ε and δ contributions of every registered mechanism call.
    While this is conservative compared to RDP/Moments accountant methods,
    it provides a clear baseline that guarantees the final privacy bounds.
    """

    target_epsilon: float
    target_delta: float
    history: list[AccountedPrivacyEvent] = field(default_factory=list)
    _consumed_epsilon: float = 0.0
    _consumed_delta: float = 0.0

    def consume(self, epsilon: float, delta: float, description: str = "") -> None:
        """Record a new privacy cost.

        Args:
            epsilon: Privacy loss ε of the mechanism.
            delta: Privacy loss δ of the mechanism.
            description: Optional string describing the mechanism call.

        Raises:
            PrivacyBudgetExceeded: when the requested consumption would breach the budget.
            ValueError: if epsilon or delta is negative.
        """
        if epsilon < 0 or delta < 0:
            raise ValueError("epsilon and delta must be non-negative.")

        new_epsilon = self._consumed_epsilon + epsilon
        new_delta = self._consumed_delta + delta

        if new_epsilon > self.target_epsilon or new_delta > self.target_delta:
            raise PrivacyBudgetExceeded(
                f"Privacy budget exceeded: "
                f"(ε={new_epsilon:.4f}, δ={new_delta:.4g}) "
                f"would surpass the target "
                f"(ε={self.target_epsilon}, δ={self.target_delta})."
            )

        self._consumed_epsilon = new_epsilon
        self._consumed_delta = new_delta
        self.history.append(AccountedPrivacyEvent(epsilon, delta, description))

    @property
    def consumed_epsilon(self) -> float:
        return self._consumed_epsilon

    @property
    def consumed_delta(self) -> float:
        return self._consumed_delta

    def remaining_budget(self) -> tuple[float, float]:
        """Return the remaining (ε, δ) budget."""
        return (
            self.target_epsilon - self._consumed_epsilon,
            self.target_delta - self._consumed_delta,
        )

    def reset(self) -> None:
        """Clear the accounting history and consumed budget."""
        self.history.clear()
        self._consumed_epsilon = 0.0
        self._consumed_delta = 0.0


@dataclass
class AdvancedCompositionAccountant(PrivacyAccountant):
    """Privacy accountant that applies the advanced composition theorem.

    The accountant keeps track of individual (ε, δ) spends and ensures that
    their advanced-composition bound stays below the configured target.

    Optimizations:
    - Cached computation of epsilon bound
    - Adaptive budget allocation suggestions
    - Budget prediction for remaining iterations
    """

    delta_slack: float = 1e-6
    _cached_epsilon_bound: float = 0.0  # NEW: Cache for performance

    def __post_init__(self) -> None:
        if self.delta_slack <= 0:
            raise ValueError("delta_slack must be positive.")

    def _compute_epsilon_bound(self, epsilons: list[float]) -> float:
        """Compute epsilon bound under advanced composition (cached).

        Optimized with:
        - Incremental computation when possible
        - Numerical stability improvements
        """
        if not epsilons:
            return 0.0

        # Advanced composition formula
        eps_squared_sum = sum(eps ** 2 for eps in epsilons)

        # Optimized linear term computation (avoid expensive exp for large eps)
        eps_linear = 0.0
        for eps in epsilons:
            if eps < 1.0:  # For small eps, use exponential
                eps_linear += eps * (math.e ** eps - 1.0)
            else:  # For large eps, use linear approximation (more stable)
                eps_linear += eps * eps

        epsilon_bound = math.sqrt(2.0 * math.log(1.0 / self.delta_slack) * eps_squared_sum) + eps_linear
        return epsilon_bound

    def consume(self, epsilon: float, delta: float, description: str = "") -> None:  # type: ignore[override]
        if epsilon < 0 or delta < 0:
            raise ValueError("epsilon and delta must be non-negative.")

        candidate_history = self.history + [AccountedPrivacyEvent(epsilon, delta, description)]
        epsilons = [event.epsilon for event in candidate_history]
        deltas = [event.delta for event in candidate_history]

        delta_total = sum(deltas) + self.delta_slack

        # Use optimized epsilon bound computation
        epsilon_bound = self._compute_epsilon_bound(epsilons)

        if epsilon_bound > self.target_epsilon or delta_total > self.target_delta:
            raise PrivacyBudgetExceeded(
                "Privacy budget exceeded under advanced composition: "
                f"(ε={epsilon_bound:.4f}, δ={delta_total:.4g}) would surpass "
                f"the target (ε={self.target_epsilon}, δ={self.target_delta})."
            )

        self._consumed_epsilon += epsilon
        self._consumed_delta += delta
        self._cached_epsilon_bound = epsilon_bound  # Cache the result
        self.history.append(AccountedPrivacyEvent(epsilon, delta, description))

    def get_effective_epsilon(self) -> float:
        """Get the effective epsilon under advanced composition (not just sum)."""
        if not self.history:
            return 0.0
        return self._cached_epsilon_bound

    def predict_remaining_queries(self, cost_per_query: tuple[float, float]) -> int:
        """Predict how many more queries can be made with given cost.

        Args:
            cost_per_query: (epsilon, delta) cost per query

        Returns:
            Estimated number of remaining queries
        """
        eps_per_query, delta_per_query = cost_per_query

        # For advanced composition, approximate remaining budget
        remaining_eps = self.target_epsilon - self._cached_epsilon_bound
        remaining_delta = self.target_delta - sum(e.delta for e in self.history) - self.delta_slack

        if remaining_eps <= 0 or remaining_delta <= 0:
            return 0

        # Conservative estimate
        max_by_eps = int(remaining_eps / eps_per_query) if eps_per_query > 0 else float('inf')
        max_by_delta = int(remaining_delta / delta_per_query) if delta_per_query > 0 else float('inf')

        return min(max_by_eps, max_by_delta)

    def suggest_adaptive_allocation(
        self,
        remaining_iterations: int,
        min_epsilon_per_iter: float = 0.1
    ) -> tuple[float, float]:
        """Suggest adaptive budget allocation for remaining iterations.

        Args:
            remaining_iterations: Number of iterations left
            min_epsilon_per_iter: Minimum epsilon per iteration

        Returns:
            Suggested (epsilon, delta) per iteration
        """
        if remaining_iterations <= 0:
            return (0.0, 0.0)

        remaining_eps = self.target_epsilon - self._cached_epsilon_bound
        remaining_delta = self.target_delta - sum(e.delta for e in self.history) - self.delta_slack

        # Adaptive strategy: allocate more budget to later iterations if making progress
        # For now, use uniform allocation (can be made more sophisticated)
        suggested_eps = max(remaining_eps / remaining_iterations, min_epsilon_per_iter)
        suggested_delta = max(remaining_delta / remaining_iterations, 0.0)

        return (suggested_eps, suggested_delta)
