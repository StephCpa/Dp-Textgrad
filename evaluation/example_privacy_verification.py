"""
Example: Privacy Verification for DP-ES

This script demonstrates how to use the privacy verification framework
to validate that DP-ES provides the promised privacy guarantees.
"""

import sys
import random
from pathlib import Path

# Add parent directory to path
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
    PrivacyAccountant,
    AdvancedCompositionAccountant,
)
from evaluation.privacy_verification import (
    NeighboringDatabaseTest,
    MembershipInferenceTest,
    NoiseDistributionTest,
    PrivacyBudgetAccuracyTest,
    PrivacyVerificationSuite,
)


# ============================================================================
# SYNTHETIC TASK: Text Quality Scoring
# ============================================================================

class SyntheticDataset:
    """Generate synthetic private dataset for testing."""

    def __init__(self, size: int = 20, seed: int = 42):
        self.rng = random.Random(seed)
        self.size = size
        self.records = self._generate_records()

    def _generate_records(self):
        """Generate synthetic text records with quality scores."""
        templates = [
            "Explain {topic} clearly and concisely.",
            "Describe {topic} in detail.",
            "Provide a summary of {topic}.",
            "Give an overview of {topic}.",
            "Analyze {topic} thoroughly.",
        ]

        topics = [
            "machine learning", "quantum computing", "climate change",
            "renewable energy", "artificial intelligence", "blockchain",
            "neural networks", "data privacy", "cloud computing", "robotics",
        ]

        records = []
        for i in range(self.size):
            template = self.rng.choice(templates)
            topic = self.rng.choice(topics)
            text = template.format(topic=topic)
            quality_score = self.rng.uniform(0.3, 1.0)
            records.append({"id": i, "text": text, "quality": quality_score})

        return records

    def get_all(self):
        return self.records.copy()


def evaluate_prompt_on_dataset(prompt: Variable, dataset: list) -> float:
    """
    Evaluate a prompt against a dataset (simulated).

    In a real scenario, this would use an LLM. Here we use a simple heuristic.
    """
    prompt_text = prompt.get_value().lower()
    score = 0.0

    # Simple heuristic based on keywords
    for record in dataset:
        record_score = record["quality"]

        # Bonus if prompt contains relevant keywords
        if "clear" in prompt_text or "concise" in prompt_text:
            record_score += 0.1
        if "detail" in prompt_text or "thorough" in prompt_text:
            record_score += 0.1

        score += record_score

    return score / len(dataset) if dataset else 0.0


# ============================================================================
# TEST 1: NEIGHBORING DATABASE TEST
# ============================================================================

def test_neighboring_databases(epsilon: float = 0.5, delta: float = 1e-5):
    """Test privacy on neighboring databases."""

    print("\n" + "🔬 TEST 1: NEIGHBORING DATABASE TEST".center(70, "="))

    dataset_obj = SyntheticDataset(size=15, seed=123)
    full_dataset = dataset_obj.get_all()

    def optimizer_factory(dataset):
        """Create and run DP-ES optimizer on given dataset."""

        prompt = Variable(
            "Answer the question.",
            role_description="instruction prompt",
            requires_grad=True
        )

        # Simple mutation for testing
        def mutation_fn(parent, iteration, rng, feedback):
            base = parent.variable.get_value()
            variations = [
                base + " Be clear.",
                base + " Be detailed.",
            ]
            return [Variable(v, role_description="prompt", requires_grad=True)
                    for v in variations[:1]]

        scorer = DPScorer(DPScorerConfig(
            clipping_value=1.0,
            noise_multiplier=None,
            epsilon=epsilon,
            delta=delta
        ))

        selector = DPSelector(DPSelectorConfig(
            select_k=2,
            epsilon=0.0,
            sensitivity=1.0
        ))

        mutation_engine = MutationEngine(
            mutation_fn=mutation_fn,
            config=MutationConfig(offspring_per_parent=1)
        )

        accountant = PrivacyAccountant(
            target_epsilon=epsilon * 5,
            target_delta=delta * 5
        )

        # Evaluation function using the provided dataset
        def eval_fn(var):
            return evaluate_prompt_on_dataset(var, dataset)

        optimizer = DPEvolutionStrategy(
            parameter=prompt,
            evaluation_fn=eval_fn,
            scorer=scorer,
            selector=selector,
            mutation_engine=mutation_engine,
            accountant=accountant,
            config=DPEvolutionConfig(
                population_size=3,
                parents_to_select=2,
                max_iterations=2,
                rng_seed=42
            )
        )

        optimizer.step()
        return prompt

    def output_extractor(prompt_var):
        return prompt_var.get_value()

    # Run test
    test = NeighboringDatabaseTest(epsilon=epsilon, delta=delta, num_trials=10)
    result = test.run_test(
        optimizer_factory=optimizer_factory,
        dataset=full_dataset,
        output_extractor=output_extractor,
        seed=42
    )

    return result


# ============================================================================
# TEST 2: MEMBERSHIP INFERENCE ATTACK
# ============================================================================

def test_membership_inference(epsilon: float = 0.5, delta: float = 1e-5):
    """Test resistance to membership inference attacks."""

    print("\n" + "🔬 TEST 2: MEMBERSHIP INFERENCE ATTACK".center(70, "="))

    dataset_obj = SyntheticDataset(size=12, seed=456)
    full_dataset = dataset_obj.get_all()

    def optimizer_factory(dataset):
        """Create optimizer for dataset."""
        prompt = Variable(
            "Provide an answer.",
            role_description="instruction",
            requires_grad=True
        )

        def mutation_fn(parent, iteration, rng, feedback):
            base = parent.variable.get_value()
            return [Variable(base + " Use examples.", role_description="prompt", requires_grad=True)]

        scorer = DPScorer(DPScorerConfig(
            clipping_value=1.0,
            epsilon=epsilon,
            delta=delta
        ))

        selector = DPSelector(DPSelectorConfig(select_k=1, epsilon=0.0))

        mutation_engine = MutationEngine(
            mutation_fn=mutation_fn,
            config=MutationConfig(offspring_per_parent=1)
        )

        accountant = PrivacyAccountant(target_epsilon=epsilon * 5, target_delta=delta * 5)

        optimizer = DPEvolutionStrategy(
            parameter=prompt,
            evaluation_fn=lambda var: evaluate_prompt_on_dataset(var, dataset),
            scorer=scorer,
            selector=selector,
            mutation_engine=mutation_engine,
            accountant=accountant,
            config=DPEvolutionConfig(
                population_size=2,
                parents_to_select=1,
                max_iterations=2,
                rng_seed=99
            )
        )

        optimizer.step()
        return prompt

    test = MembershipInferenceTest(epsilon=epsilon, delta=delta)
    result = test.run_test(
        optimizer_factory=optimizer_factory,
        dataset=full_dataset,
        output_extractor=lambda p: p.get_value(),
        num_trials=10,
        seed=789
    )

    return result


# ============================================================================
# TEST 3: NOISE DISTRIBUTION VERIFICATION
# ============================================================================

def test_noise_distribution(epsilon: float = 0.5, delta: float = 1e-5):
    """Verify noise distribution properties."""

    print("\n" + "🔬 TEST 3: NOISE DISTRIBUTION VERIFICATION".center(70, "="))

    # Generate test scores
    import numpy as np
    np.random.seed(42)
    true_scores = list(np.random.uniform(0.3, 0.9, 20))

    def scorer_factory():
        return DPScorer(DPScorerConfig(
            clipping_value=1.0,
            epsilon=epsilon,
            delta=delta
        ))

    test = NoiseDistributionTest(epsilon=epsilon, delta=delta, sensitivity=1.0)
    result = test.run_test(
        scorer_factory=scorer_factory,
        true_scores=true_scores,
        num_trials=100,
        seed=42
    )

    return result


# ============================================================================
# TEST 4: PRIVACY BUDGET TRACKING
# ============================================================================

def test_budget_tracking():
    """Test privacy budget accounting accuracy."""

    print("\n" + "🔬 TEST 4: PRIVACY BUDGET TRACKING".center(70, "="))

    # Test basic accountant
    print("\n--- Basic Composition Accountant ---")
    accountant_basic = PrivacyAccountant(target_epsilon=5.0, target_delta=1e-4)

    operations = [
        (0.5, 1e-5, "scoring_round_1"),
        (0.5, 1e-5, "scoring_round_2"),
        (0.1, 0.0, "selection_round_1"),
        (0.5, 1e-5, "scoring_round_3"),
    ]

    test_basic = PrivacyBudgetAccuracyTest()
    result_basic = test_basic.run_test(accountant_basic, operations)

    # Test advanced accountant
    print("\n--- Advanced Composition Accountant ---")
    accountant_advanced = AdvancedCompositionAccountant(
        target_epsilon=5.0,
        target_delta=1e-4,
        delta_slack=1e-6
    )

    test_advanced = PrivacyBudgetAccuracyTest()
    result_advanced = test_advanced.run_test(accountant_advanced, operations)

    # Compare
    improvement = result_basic.metrics["consumed_epsilon"] - result_advanced.metrics["consumed_epsilon"]
    print(f"\n✓ Privacy improvement with advanced composition: Δε = {improvement:.3f}")

    return result_basic, result_advanced


# ============================================================================
# MAIN: RUN ALL TESTS
# ============================================================================

def main():
    """Run complete privacy verification suite."""

    print("\n" + "="*70)
    print("DIFFERENTIAL PRIVACY VERIFICATION SUITE FOR DP-ES")
    print("="*70)
    print("\nThis suite empirically verifies that DP-ES provides")
    print("the promised differential privacy guarantees.\n")

    # Create verification suite
    suite = PrivacyVerificationSuite(output_dir="privacy_verification_results")

    # Privacy parameters for testing
    EPSILON = 0.5
    DELTA = 1e-5

    try:
        # Test 1: Neighboring databases
        result1 = test_neighboring_databases(epsilon=EPSILON, delta=DELTA)
        suite.add_result(result1)

    except Exception as e:
        print(f"\n⚠️  Test 1 failed with error: {e}")

    try:
        # Test 2: Membership inference
        result2 = test_membership_inference(epsilon=EPSILON, delta=DELTA)
        suite.add_result(result2)

    except Exception as e:
        print(f"\n⚠️  Test 2 failed with error: {e}")

    try:
        # Test 3: Noise distribution
        result3 = test_noise_distribution(epsilon=EPSILON, delta=DELTA)
        suite.add_result(result3)

    except Exception as e:
        print(f"\n⚠️  Test 3 failed with error: {e}")

    try:
        # Test 4: Budget tracking
        result4_basic, result4_advanced = test_budget_tracking()
        suite.add_result(result4_basic)
        suite.add_result(result4_advanced)

    except Exception as e:
        print(f"\n⚠️  Test 4 failed with error: {e}")

    # Generate final report
    report = suite.generate_report()
    print(report)

    print("\n" + "="*70)
    print("✓ VERIFICATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {suite.output_dir}/")
    print("  - verification_report.txt (human-readable)")
    print("  - verification_results.json (machine-readable)")
    print("\n")


if __name__ == "__main__":
    main()
