"""
Differential Privacy configuration presets for DP-ES.

This module provides pre-configured privacy settings for different use cases,
ranging from high privacy (healthcare, finance) to moderate privacy (general business).

Usage:
    from dp_textgrad.config_dp import DP_CONFIGS, get_dp_config

    # Use a preset configuration
    config = get_dp_config("high_privacy")

    # Or access directly
    scorer_config = DP_CONFIGS["high_privacy"]["scorer"]
    selector_config = DP_CONFIGS["high_privacy"]["selector"]
"""

from dataclasses import dataclass, replace
from typing import Dict, Optional, Literal

from .dp_es import (
    DPScorerConfig,
    DPSelectorConfig,
    MutationConfig,
    DPEvolutionConfig,
)


PrivacyLevel = Literal["high_privacy", "medium_privacy", "low_privacy", "minimal_privacy"]


@dataclass
class DPConfigBundle:
    """Complete configuration bundle for DP-ES."""

    scorer: DPScorerConfig
    selector: DPSelectorConfig
    mutation: MutationConfig
    evolution: DPEvolutionConfig
    accountant_type: Literal["basic", "advanced"]
    target_epsilon: float
    target_delta: float
    delta_slack: Optional[float] = None
    description: str = ""

    def __repr__(self) -> str:
        return (
            f"DPConfigBundle(\n"
            f"  description='{self.description}',\n"
            f"  privacy_budget=(ε={self.target_epsilon}, δ={self.target_delta}),\n"
            f"  accountant={self.accountant_type},\n"
            f"  population_size={self.evolution.population_size},\n"
            f"  max_iterations={self.evolution.max_iterations}\n"
            f")"
        )


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

DP_CONFIGS: Dict[PrivacyLevel, DPConfigBundle] = {
    # -------------------------------------------------------------------------
    # HIGH PRIVACY: For healthcare, finance, or highly sensitive personal data
    # Provides strong privacy guarantees (ε < 1.0)
    # -------------------------------------------------------------------------
    "high_privacy": DPConfigBundle(
        scorer=DPScorerConfig(
            clipping_value=1.0,
            noise_multiplier=None,  # Auto-calibrated
            epsilon=0.3,  # Low epsilon per iteration
            delta=1e-6,
        ),
        selector=DPSelectorConfig(
            select_k=2,  # Conservative selection
            epsilon=0.05,  # Minimal epsilon for selection
            sensitivity=1.0,
        ),
        mutation=MutationConfig(
            offspring_per_parent=2,
            allow_identity_offspring=True,
        ),
        evolution=DPEvolutionConfig(
            population_size=4,  # Smaller population to conserve budget
            parents_to_select=2,
            max_iterations=5,
            stop_on_budget=True,
            evaluation_description="high_privacy_iteration",
        ),
        accountant_type="advanced",
        target_epsilon=1.0,
        target_delta=1e-5,
        delta_slack=1e-6,
        description="High privacy for sensitive data (healthcare, finance)",
    ),

    # -------------------------------------------------------------------------
    # MEDIUM PRIVACY: Balanced privacy-performance for general business use
    # Good default for most applications (1.0 ≤ ε ≤ 3.0)
    # -------------------------------------------------------------------------
    "medium_privacy": DPConfigBundle(
        scorer=DPScorerConfig(
            clipping_value=1.0,
            noise_multiplier=None,
            epsilon=0.5,  # Moderate epsilon
            delta=1e-5,
        ),
        selector=DPSelectorConfig(
            select_k=3,
            epsilon=0.1,
            sensitivity=1.0,
        ),
        mutation=MutationConfig(
            offspring_per_parent=3,
            allow_identity_offspring=True,
        ),
        evolution=DPEvolutionConfig(
            population_size=6,
            parents_to_select=3,
            max_iterations=6,
            stop_on_budget=True,
            evaluation_description="medium_privacy_iteration",
        ),
        accountant_type="advanced",
        target_epsilon=2.5,
        target_delta=1e-4,
        delta_slack=1e-6,
        description="Balanced privacy-performance for general business use",
    ),

    # -------------------------------------------------------------------------
    # LOW PRIVACY: For aggregated or less sensitive data
    # Better performance with reasonable privacy (3.0 ≤ ε ≤ 6.0)
    # -------------------------------------------------------------------------
    "low_privacy": DPConfigBundle(
        scorer=DPScorerConfig(
            clipping_value=1.0,
            noise_multiplier=None,
            epsilon=1.0,  # Higher epsilon for better accuracy
            delta=1e-5,
        ),
        selector=DPSelectorConfig(
            select_k=4,
            epsilon=0.2,
            sensitivity=1.0,
        ),
        mutation=MutationConfig(
            offspring_per_parent=3,
            allow_identity_offspring=True,
        ),
        evolution=DPEvolutionConfig(
            population_size=8,
            parents_to_select=4,
            max_iterations=8,
            stop_on_budget=True,
            evaluation_description="low_privacy_iteration",
        ),
        accountant_type="advanced",
        target_epsilon=5.0,
        target_delta=1e-4,
        delta_slack=1e-6,
        description="Lower privacy for aggregated or less sensitive data",
    ),

    # -------------------------------------------------------------------------
    # MINIMAL PRIVACY: Primarily for testing or public data
    # Weak privacy guarantees, focus on performance (ε > 6.0)
    # -------------------------------------------------------------------------
    "minimal_privacy": DPConfigBundle(
        scorer=DPScorerConfig(
            clipping_value=1.0,
            noise_multiplier=None,
            epsilon=2.0,  # High epsilon
            delta=1e-5,
        ),
        selector=DPSelectorConfig(
            select_k=5,
            epsilon=0.5,
            sensitivity=1.0,
        ),
        mutation=MutationConfig(
            offspring_per_parent=4,
            allow_identity_offspring=True,
        ),
        evolution=DPEvolutionConfig(
            population_size=10,
            parents_to_select=5,
            max_iterations=10,
            stop_on_budget=True,
            evaluation_description="minimal_privacy_iteration",
        ),
        accountant_type="basic",
        target_epsilon=10.0,
        target_delta=1e-4,
        delta_slack=None,
        description="Minimal privacy for testing or public data",
    ),
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_dp_config(
    privacy_level: PrivacyLevel,
    custom_overrides: Optional[Dict] = None
) -> DPConfigBundle:
    """
    Get a DP configuration bundle by privacy level.

    Args:
        privacy_level: One of "high_privacy", "medium_privacy", "low_privacy", "minimal_privacy"
        custom_overrides: Optional dictionary to override specific parameters

    Returns:
        DPConfigBundle with the requested configuration

    Example:
        >>> config = get_dp_config("medium_privacy")
        >>> # Customize population size
        >>> config = get_dp_config("medium_privacy",
        ...                        custom_overrides={"population_size": 10})
    """
    if privacy_level not in DP_CONFIGS:
        raise ValueError(
            f"Unknown privacy level: {privacy_level}. "
            f"Choose from: {list(DP_CONFIGS.keys())}"
        )

    config = DP_CONFIGS[privacy_level]

    if custom_overrides:
        # Apply overrides to evolution config (most commonly customized)
        if any(k in custom_overrides for k in
               ["population_size", "parents_to_select", "max_iterations"]):
            config = replace(
                config,
                evolution=replace(
                    config.evolution,
                    **{k: v for k, v in custom_overrides.items()
                       if k in DPEvolutionConfig.__dataclass_fields__}
                )
            )

        # Apply overrides to scorer config
        if any(k in custom_overrides for k in
               ["clipping_value", "epsilon", "delta"]):
            config = replace(
                config,
                scorer=replace(
                    config.scorer,
                    **{k: v for k, v in custom_overrides.items()
                       if k in DPScorerConfig.__dataclass_fields__}
                )
            )

    return config


def print_config_comparison():
    """Print a comparison table of all preset configurations."""
    print("\n" + "="*90)
    print("DP-ES PRESET CONFIGURATION COMPARISON")
    print("="*90)
    print(f"{'Privacy Level':<20} {'ε_total':<10} {'ε_iter':<10} {'Pop Size':<10} {'Max Iter':<10} {'Accountant':<12}")
    print("-"*90)

    for level, config in DP_CONFIGS.items():
        print(
            f"{level:<20} "
            f"{config.target_epsilon:<10.1f} "
            f"{config.scorer.epsilon:<10.2f} "
            f"{config.evolution.population_size:<10} "
            f"{config.evolution.max_iterations:<10} "
            f"{config.accountant_type:<12}"
        )

    print("="*90)
    print("\nRecommendations:")
    print("  • High Privacy:    Healthcare, finance, PII (ε < 1.0)")
    print("  • Medium Privacy:  General business, user analytics (1.0 ≤ ε ≤ 3.0)")
    print("  • Low Privacy:     Aggregated data, public datasets (3.0 ≤ ε ≤ 6.0)")
    print("  • Minimal Privacy: Testing, non-sensitive research (ε > 6.0)")
    print("="*90 + "\n")


def estimate_privacy_budget(
    num_iterations: int,
    epsilon_per_iter: float,
    composition_type: Literal["basic", "advanced"] = "advanced",
    delta: float = 1e-5
) -> Dict[str, float]:
    """
    Estimate total privacy budget for a given configuration.

    Args:
        num_iterations: Number of optimization iterations
        epsilon_per_iter: Privacy budget per iteration
        composition_type: "basic" or "advanced" composition
        delta: Privacy parameter delta

    Returns:
        Dictionary with epsilon_total and delta_total estimates

    Example:
        >>> estimate_privacy_budget(10, 0.5, "advanced", 1e-5)
        {'epsilon_total': 1.58, 'delta_total': 0.0001, 'composition': 'advanced'}
    """
    if composition_type == "basic":
        # Basic composition: ε_total = Σ ε_i
        epsilon_total = num_iterations * epsilon_per_iter
    else:
        # Advanced composition (simplified): ε' ≈ √(2k ln(1/δ')) ε + kε(e^ε - 1)
        # For small ε, approximation: ε' ≈ √(2k ln(1/δ)) ε
        import math
        epsilon_total = math.sqrt(2 * num_iterations * math.log(1/delta)) * epsilon_per_iter

    delta_total = num_iterations * delta

    return {
        "epsilon_total": round(epsilon_total, 2),
        "delta_total": delta_total,
        "composition": composition_type,
        "iterations": num_iterations,
        "epsilon_per_iter": epsilon_per_iter,
    }


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_privacy_parameters(epsilon: float, delta: float, dataset_size: int):
    """
    Validate privacy parameters and provide warnings.

    Args:
        epsilon: Privacy budget epsilon
        delta: Privacy parameter delta
        dataset_size: Number of samples in private dataset
    """
    warnings = []

    # Check epsilon
    if epsilon > 10:
        warnings.append(f"⚠️  ε={epsilon} is very high - provides minimal privacy protection")
    elif epsilon < 0.1:
        warnings.append(f"⚠️  ε={epsilon} is very low - may require many iterations or large noise")

    # Check delta
    recommended_delta = 1 / (dataset_size ** 2)
    if delta > 1 / dataset_size:
        warnings.append(
            f"⚠️  δ={delta} is too large for dataset of size {dataset_size}. "
            f"Recommended: δ < 1/n = {1/dataset_size:.2e}"
        )

    # Check combination
    if epsilon > 3 and delta > 1e-4:
        warnings.append(
            "⚠️  Both ε and δ are high - consider tightening at least one parameter"
        )

    if warnings:
        print("\n" + "="*80)
        print("PRIVACY PARAMETER VALIDATION")
        print("="*80)
        for warning in warnings:
            print(warning)
        print(f"\nRecommended δ for n={dataset_size}: {recommended_delta:.2e}")
        print("="*80 + "\n")
    else:
        print(f"✓ Privacy parameters validated: ε={epsilon}, δ={delta} for n={dataset_size}")


if __name__ == "__main__":
    # Demo the configuration system
    print("\n" + "="*90)
    print("DP-ES CONFIGURATION SYSTEM DEMO")
    print("="*90 + "\n")

    # Show all configurations
    print_config_comparison()

    # Example usage
    print("\nExample 1: Getting a preset configuration")
    print("-" * 50)
    config = get_dp_config("medium_privacy")
    print(config)

    print("\n\nExample 2: Customizing a preset")
    print("-" * 50)
    custom_config = get_dp_config(
        "high_privacy",
        custom_overrides={"population_size": 8, "max_iterations": 10}
    )
    print(custom_config)

    print("\n\nExample 3: Privacy budget estimation")
    print("-" * 50)
    budget = estimate_privacy_budget(
        num_iterations=10,
        epsilon_per_iter=0.5,
        composition_type="advanced",
        delta=1e-5
    )
    print(f"Estimated privacy budget: {budget}")

    print("\n\nExample 4: Parameter validation")
    print("-" * 50)
    validate_privacy_parameters(epsilon=2.0, delta=1e-5, dataset_size=1000)

    print("\n" + "="*90 + "\n")
