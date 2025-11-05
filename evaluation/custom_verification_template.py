"""
Template: Custom Privacy Verification for Your Task

This template shows how to adapt the privacy verification framework
to your specific DP-ES use case. Replace the placeholder sections
with your actual task implementation.

Usage:
    1. Copy this file to your project
    2. Replace TODOs with your task logic
    3. Run: python custom_verification_template.py
"""

import sys
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from dp_textgrad import Variable
from dp_textgrad.dp_es import (
    DPEvolutionStrategy,
    DPEvolutionConfig,
    DPScorer,
    DPScorerConfig,
    DPSelector,
    DPSelectorConfig,
    MutationEngine,
    MutationConfig,
    AdvancedCompositionAccountant,
)
from evaluation.privacy_verification import (
    NeighboringDatabaseTest,
    MembershipInferenceTest,
    PrivacyVerificationSuite,
)


# ============================================================================
# SECTION 1: DEFINE YOUR TASK
# ============================================================================

def load_your_dataset():
    """
    TODO: Load your actual dataset structure (without private data!)

    For privacy testing, you can use:
    - Synthetic data that mimics your real data structure
    - Anonymized/public subset of your data
    - Generated samples with similar properties

    Returns:
        List of records in your dataset format
    """
    # Example: Replace with your data loading logic
    dataset = [
        {"id": i, "text": f"Sample {i}", "label": i % 3}
        for i in range(20)  # Adjust size
    ]
    return dataset


def evaluate_on_dataset(variable: Variable, dataset: list) -> float:
    """
    TODO: Implement your evaluation function.

    This function should:
    - Take a Variable (e.g., optimized prompt, solution, code)
    - Evaluate it against your dataset
    - Return a single numeric score (higher = better)

    IMPORTANT: Only return aggregate scores, no individual record info!

    Args:
        variable: The Variable being optimized
        dataset: Your dataset records

    Returns:
        Aggregate score (e.g., accuracy, F1, quality metric)
    """
    # Example: Replace with your evaluation logic
    prompt = variable.get_value()
    score = 0.0

    for record in dataset:
        # TODO: Your evaluation logic here
        # Example: simple heuristic
        if len(prompt) > 10:
            score += 0.5
        if record["label"] == 0:
            score += 0.3

    return score / len(dataset) if dataset else 0.0


# ============================================================================
# SECTION 2: DEFINE YOUR DP-ES CONFIGURATION
# ============================================================================

def create_dp_optimizer(dataset, epsilon=0.5, delta=1e-5, seed=42):
    """
    TODO: Configure your DP-ES optimizer.

    Adjust these to match your task:
    - Initial variable value
    - Mutation strategy
    - Population size
    - Privacy budget

    Args:
        dataset: Your dataset
        epsilon: Privacy budget per iteration
        delta: Privacy parameter
        seed: Random seed

    Returns:
        Configured DPEvolutionStrategy instance
    """

    # 1. Define initial variable
    # TODO: Customize initial value and role description
    initial_variable = Variable(
        value="Your initial prompt/solution/code here",
        role_description="description of what this variable represents",
        requires_grad=True
    )

    # 2. Define mutation function
    # TODO: Implement your mutation strategy
    def mutation_function(parent_candidate, iteration, rng, feedback):
        """Generate variations of the parent."""
        parent_value = parent_candidate.variable.get_value()

        # Example: Simple variations (replace with your logic)
        variations = [
            parent_value + " [variation 1]",
            parent_value + " [variation 2]",
        ]

        return [
            Variable(v, role_description=initial_variable.get_role_description(), requires_grad=True)
            for v in variations
        ]

    # 3. Configure DP components
    scorer = DPScorer(DPScorerConfig(
        clipping_value=1.0,  # TODO: Adjust based on your score range
        noise_multiplier=None,  # Auto-calibrate
        epsilon=epsilon,
        delta=delta
    ))

    selector = DPSelector(DPSelectorConfig(
        select_k=2,  # TODO: Adjust based on population size
        epsilon=0.0,  # No additional epsilon for selection
        sensitivity=1.0
    ))

    mutation_engine = MutationEngine(
        mutation_fn=mutation_function,
        config=MutationConfig(
            offspring_per_parent=2,  # TODO: Adjust
            allow_identity_offspring=True
        )
    )

    accountant = AdvancedCompositionAccountant(
        target_epsilon=epsilon * 10,  # TODO: Set total budget
        target_delta=delta * 10
    )

    # 4. Create optimizer
    optimizer = DPEvolutionStrategy(
        parameter=initial_variable,
        evaluation_fn=lambda var: evaluate_on_dataset(var, dataset),
        scorer=scorer,
        selector=selector,
        mutation_engine=mutation_engine,
        accountant=accountant,
        config=DPEvolutionConfig(
            population_size=4,  # TODO: Adjust
            parents_to_select=2,
            max_iterations=3,  # TODO: Adjust
            rng_seed=seed
        )
    )

    return optimizer


# ============================================================================
# SECTION 3: RUN PRIVACY VERIFICATION
# ============================================================================

def run_neighboring_database_test():
    """Test privacy on neighboring databases."""

    print("\n" + "="*70)
    print("NEIGHBORING DATABASE TEST - YOUR TASK")
    print("="*70)

    dataset = load_your_dataset()
    epsilon, delta = 0.5, 1e-5

    def optimizer_factory(data):
        """Create optimizer for given dataset."""
        opt = create_dp_optimizer(data, epsilon=epsilon, delta=delta)
        opt.step()
        return opt.parameter  # Return the optimized variable

    def output_extractor(variable):
        """Extract comparable output."""
        return variable.get_value()

    # Run test
    test = NeighboringDatabaseTest(
        epsilon=epsilon,
        delta=delta,
        num_trials=5  # TODO: Increase for production
    )

    result = test.run_test(
        optimizer_factory=optimizer_factory,
        dataset=dataset,
        output_extractor=output_extractor,
        seed=42
    )

    return result


def run_membership_inference_test():
    """Test resistance to membership inference."""

    print("\n" + "="*70)
    print("MEMBERSHIP INFERENCE TEST - YOUR TASK")
    print("="*70)

    dataset = load_your_dataset()
    epsilon, delta = 0.5, 1e-5

    def optimizer_factory(data):
        opt = create_dp_optimizer(data, epsilon=epsilon, delta=delta)
        opt.step()
        return opt.parameter

    test = MembershipInferenceTest(epsilon=epsilon, delta=delta)

    result = test.run_test(
        optimizer_factory=optimizer_factory,
        dataset=dataset,
        output_extractor=lambda v: v.get_value(),
        num_trials=5,  # TODO: Increase for production
        seed=123
    )

    return result


# ============================================================================
# MAIN: RUN YOUR CUSTOM VERIFICATION
# ============================================================================

def main():
    """Run privacy verification on your custom task."""

    print("\n" + "="*70)
    print("CUSTOM PRIVACY VERIFICATION")
    print("="*70)
    print("\nTesting privacy guarantees for YOUR task...")
    print("="*70)

    # Create verification suite
    suite = PrivacyVerificationSuite(
        output_dir="custom_privacy_results"
    )

    # Run tests
    try:
        print("\n[1/2] Running Neighboring Database Test...")
        result1 = run_neighboring_database_test()
        suite.add_result(result1)
    except Exception as e:
        print(f"\n⚠️  Test 1 failed: {e}")

    try:
        print("\n[2/2] Running Membership Inference Test...")
        result2 = run_membership_inference_test()
        suite.add_result(result2)
    except Exception as e:
        print(f"\n⚠️  Test 2 failed: {e}")

    # Generate report
    report = suite.generate_report()
    print(report)

    print("\n" + "="*70)
    print("✓ CUSTOM VERIFICATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {suite.output_dir}/")
    print("\n💡 NEXT STEPS:")
    print("   1. Review the verification report")
    print("   2. Adjust privacy parameters if needed")
    print("   3. Run with more trials for higher confidence")
    print("   4. Generate visualizations:")
    print("      python evaluation/privacy_visualization.py")
    print("\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CUSTOM PRIVACY VERIFICATION TEMPLATE")
    print("="*70)
    print("\n⚠️  IMPORTANT: This is a TEMPLATE file!")
    print("\nBefore running:")
    print("  1. Implement load_your_dataset()")
    print("  2. Implement evaluate_on_dataset()")
    print("  3. Customize mutation_function()")
    print("  4. Adjust privacy parameters")
    print("\nSearchfor 'TODO' comments in this file for guidance.")
    print("="*70)

    # Uncomment to run (after implementing TODOs)
    # main()

    print("\n✓ Template loaded successfully!")
    print("  Copy this file and customize it for your task.\n")
