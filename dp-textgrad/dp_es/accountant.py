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
    """

    delta_slack: float = 1e-6

    def __post_init__(self) -> None:
        if self.delta_slack <= 0:
            raise ValueError("delta_slack must be positive.")

    def consume(self, epsilon: float, delta: float, description: str = "") -> None:  # type: ignore[override]
        if epsilon < 0 or delta < 0:
            raise ValueError("epsilon and delta must be non-negative.")

        candidate_history = self.history + [AccountedPrivacyEvent(epsilon, delta, description)]
        epsilons = [event.epsilon for event in candidate_history]
        deltas = [event.delta for event in candidate_history]

        delta_total = sum(deltas) + self.delta_slack

        eps_squared_sum = sum(eps ** 2 for eps in epsilons)
        eps_linear = sum(eps * max(math.e ** eps - 1.0, eps) for eps in epsilons)
        epsilon_bound = math.sqrt(2.0 * math.log(1.0 / self.delta_slack) * eps_squared_sum) + eps_linear

        if epsilon_bound > self.target_epsilon or delta_total > self.target_delta:
            raise PrivacyBudgetExceeded(
                "Privacy budget exceeded under advanced composition: "
                f"(ε={epsilon_bound:.4f}, δ={delta_total:.4g}) would surpass "
                f"the target (ε={self.target_epsilon}, δ={self.target_delta})."
            )

        self._consumed_epsilon += epsilon
        self._consumed_delta += delta
        self.history.append(AccountedPrivacyEvent(epsilon, delta, description))
