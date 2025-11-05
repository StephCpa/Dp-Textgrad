"""
DP-ES Quick Start Example

This example demonstrates the easiest way to get started with
Differential Privacy Evolution Strategy using preset configurations.
"""

import dp_textgrad as tg
from dp_textgrad import Variable, BlackboxLLM
from dp_textgrad.config_dp import get_dp_config
from dp_textgrad.dp_es import (
    DPEvolutionStrategy,
    PrivacyAccountant,
    AdvancedCompositionAccountant,
    DPScorer,
    DPSelector,
    MutationEngine,
)

# Set your API key
# import os
# os.environ["OPENAI_API_KEY"] = "your-api-key"

# Set the backward engine
tg.set_backward_engine("gpt-4o-mini", override=True)


def main():
    print("="*70)
    print("DP-ES QUICK START EXAMPLE")
    print("="*70)

    # ========================================================================
    # 1. GET A PRESET CONFIGURATION
    # ========================================================================
    # Choose from: "high_privacy", "medium_privacy", "low_privacy", "minimal_privacy"

    print("\n📋 Loading preset configuration...")
    config = get_dp_config("medium_privacy")
    print(config)

    # ========================================================================
    # 2. DEFINE YOUR TASK
    # ========================================================================

    print("\n📝 Setting up task...")

    # Define initial variable to optimize
    initial_prompt = "Solve this problem."
    target_variable = Variable(
        value=initial_prompt,
        role_description="instruction for problem solving",
        requires_grad=True
    )

    # Define evaluation function (this uses your private data!)
    model = BlackboxLLM("gpt-4o-mini")

    def evaluate_prompt(variable: Variable) -> float:
        """Evaluate the prompt - replace with your actual task."""
        prompt = variable.get_value()

        # Example: simple heuristic scoring
        score = 0.0
        if len(prompt) > 20:
            score += 0.3
        if "explain" in prompt.lower() or "step" in prompt.lower():
            score += 0.4
        if "clear" in prompt.lower() or "detailed" in prompt.lower():
            score += 0.3

        return min(score, 1.0)

    print(f"  Initial prompt: '{initial_prompt}'")
    print(f"  Initial score: {evaluate_prompt(target_variable):.3f}")

    # ========================================================================
    # 3. CREATE DP COMPONENTS FROM CONFIG
    # ========================================================================

    print("\n🔧 Creating DP-ES components...")

    # Create scorer
    scorer = DPScorer(config.scorer)

    # Create selector
    selector = DPSelector(config.selector)

    # Create mutation engine
    def mutation_function(parent_candidate, iteration, rng, feedback):
        """Generate variations of the parent."""
        parent_text = parent_candidate.variable.get_value()

        # Simple variations (replace with your own logic)
        variations = [
            parent_text + " Be clear and detailed.",
            parent_text + " Explain step by step.",
        ]

        return [
            Variable(v, role_description="prompt", requires_grad=True)
            for v in variations
        ]

    mutation_engine = MutationEngine(
        mutation_fn=mutation_function,
        config=config.mutation
    )

    # Create privacy accountant
    if config.accountant_type == "advanced":
        accountant = AdvancedCompositionAccountant(
            target_epsilon=config.target_epsilon,
            target_delta=config.target_delta,
            delta_slack=config.delta_slack
        )
    else:
        accountant = PrivacyAccountant(
            target_epsilon=config.target_epsilon,
            target_delta=config.target_delta
        )

    print(f"✓ Privacy budget: ε={config.target_epsilon}, δ={config.target_delta}")

    # ========================================================================
    # 4. RUN DP-ES OPTIMIZATION
    # ========================================================================

    print("\n🚀 Running DP-ES optimization...\n")

    optimizer = DPEvolutionStrategy(
        parameter=target_variable,
        evaluation_fn=evaluate_prompt,
        scorer=scorer,
        selector=selector,
        mutation_engine=mutation_engine,
        accountant=accountant,
        config=config.evolution
    )

    # Run optimization
    optimizer.step()

    # ========================================================================
    # 5. CHECK RESULTS
    # ========================================================================

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    final_score = evaluate_prompt(target_variable)
    print(f"\n✓ Optimized prompt: '{target_variable.get_value()}'")
    print(f"✓ Final score: {final_score:.3f}")
    print(f"✓ Improvement: {final_score - evaluate_prompt(Variable(initial_prompt, role_description='test')):.3f}")

    print(f"\n🔒 Privacy consumed:")
    print(f"   ε = {accountant.consumed_epsilon:.3f} / {config.target_epsilon}")
    print(f"   δ = {accountant.consumed_delta:.2e} / {config.target_delta}")

    remaining_eps, remaining_delta = accountant.remaining_budget()
    print(f"\n📊 Privacy remaining:")
    print(f"   ε = {remaining_eps:.3f}")
    print(f"   δ = {remaining_delta:.2e}")

    print("\n" + "="*70)
    print("✓ Optimization complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    # First, show all available configurations
    print("\nAvailable preset configurations:")
    print("-"*70)
    tg.print_config_comparison()

    # Run the example
    main()

    # Show privacy budget estimation
    print("\n💡 TIP: Estimate privacy budget before running:")
    print("-"*70)
    budget_estimate = tg.estimate_privacy_budget(
        num_iterations=6,
        epsilon_per_iter=0.5,
        composition_type="advanced",
        delta=1e-5
    )
    print(f"For 6 iterations with ε=0.5 per iteration:")
    print(f"  Estimated total ε: {budget_estimate['epsilon_total']}")
    print(f"  Estimated total δ: {budget_estimate['delta_total']:.2e}")
    print(f"  Composition: {budget_estimate['composition']}")

    print("\n✓ Done! Next steps:")
    print("  1. Replace evaluate_prompt() with your actual task")
    print("  2. Implement custom mutation_function() if needed")
    print("  3. Try different privacy levels: 'high_privacy', 'low_privacy', etc.")
    print("  4. See Tutorial-DP-Evolution-Strategy.ipynb for more examples\n")
