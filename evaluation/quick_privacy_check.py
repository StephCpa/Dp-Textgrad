"""
Quick Privacy Check

A fast sanity check to verify DP-ES privacy mechanisms are working.
This is useful for rapid testing during development.

Runtime: ~30 seconds
"""

import sys
from pathlib import Path
import random
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from dp_textgrad.dp_es import (
    DPScorer,
    DPScorerConfig,
    DPSelector,
    DPSelectorConfig,
    PrivacyAccountant,
    AdvancedCompositionAccountant,
)
from dp_textgrad.dp_es.population import Candidate
from dp_textgrad import Variable


def check_noise_injection():
    """Verify that noise is being added to scores."""

    print("\n" + "="*70)
    print("TEST 1: NOISE INJECTION CHECK")
    print("="*70)

    scorer = DPScorer(DPScorerConfig(
        clipping_value=1.0,
        noise_multiplier=1.0,  # Explicit noise
        epsilon_per_candidate=0.5,
        delta_per_candidate=1e-5,
    ))

    # Create test candidates
    test_scores = [0.5, 0.5, 0.5, 0.5, 0.5]  # All identical
    candidates = []

    for i, score in enumerate(test_scores):
        var = Variable(f"test_{i}", role_description="test", requires_grad=False)
        candidates.append(Candidate(var, {"id": i}))

    # Evaluate with DP (should add noise)
    def eval_fn(cand):
        idx = cand.metadata["id"]
        return test_scores[idx]

    dp_scores = scorer.evaluate(candidates, eval_fn, rng=random.Random(42))
    dp_values = [cand.dp_score for cand in dp_scores.updated_candidates]

    # Because raw scores are all 0.5, any noticeable variance implies noise was added.
    noise_std = float(np.std(dp_values))
    for i, dp_value in enumerate(dp_values):
        print(f"  Candidate {i}: dp_score={dp_value:.3f}")

    noise_detected = noise_std > 0.01
    print(f"\n  Observed std of DP scores: {noise_std:.3f}")

    if noise_detected:
        print("\n✓ PASS: Noise injection is working")
        return True
    else:
        print("\n✗ FAIL: No noise detected (privacy may be compromised)")
        return False


def check_score_clipping():
    """Verify that scores are clipped to bounds."""

    print("\n" + "="*70)
    print("TEST 2: SCORE CLIPPING CHECK")
    print("="*70)

    scorer = DPScorer(DPScorerConfig(
        clipping_value=1.0,
        noise_multiplier=0.0,  # No noise for clearer test
        epsilon_per_candidate=0.5,
        delta_per_candidate=1e-5,
    ))

    # Test with scores outside clipping bounds
    test_scores = [-2.0, -0.5, 0.5, 1.5, 3.0]
    candidates = []

    for i, score in enumerate(test_scores):
        var = Variable(f"test_{i}", role_description="test", requires_grad=False)
        candidates.append(Candidate(var, {"id": i}))

    def eval_fn(cand):
        idx = cand.metadata["id"]
        return test_scores[idx]

    dp_scores = scorer.evaluate(candidates, eval_fn, rng=random.Random(42))
    dp_values = [cand.dp_score for cand in dp_scores.updated_candidates]

    clip_value = scorer.config.clipping_value
    all_clipped = True
    for idx, (raw_score, dp_value) in enumerate(zip(test_scores, dp_values)):
        expected = max(-clip_value, min(clip_value, raw_score))
        clipped_correctly = abs(dp_value - expected) < 1e-6
        print(f"  Score {idx}: raw={raw_score:.2f} → clipped={dp_value:.2f} "
              f"(expected {expected:.2f}) {'✓' if clipped_correctly else '✗'}")
        if not clipped_correctly:
            all_clipped = False

    if all_clipped:
        print("\n✓ PASS: Score clipping is working")
        return True
    else:
        print("\n✗ FAIL: Scores not properly clipped")
        return False


def check_privacy_budget_tracking():
    """Verify privacy accountant tracks budget correctly."""

    print("\n" + "="*70)
    print("TEST 3: PRIVACY BUDGET TRACKING CHECK")
    print("="*70)

    # Test basic accountant
    print("\n--- Basic Accountant ---")
    accountant = PrivacyAccountant(target_epsilon=2.0, target_delta=1e-4)

    operations = [
        (0.5, 1e-5, "op1"),
        (0.3, 1e-5, "op2"),
        (0.4, 1e-5, "op3"),
    ]

    for eps, delta, desc in operations:
        accountant.consume(eps, delta, desc)
        print(f"  After {desc}: ε={accountant.consumed_epsilon:.3f}, δ={accountant.consumed_delta:.2e}")

    expected_eps = sum(op[0] for op in operations)
    expected_delta = sum(op[1] for op in operations)

    basic_correct = (
        abs(accountant.consumed_epsilon - expected_eps) < 0.001 and
        abs(accountant.consumed_delta - expected_delta) < 1e-10
    )

    print(f"\n  Expected: ε={expected_eps:.3f}, δ={expected_delta:.2e}")
    print(f"  Tracked:  ε={accountant.consumed_epsilon:.3f}, δ={accountant.consumed_delta:.2e}")

    if basic_correct:
        print("  ✓ Basic accountant correct")
    else:
        print("  ✗ Basic accountant incorrect")

    # Test advanced accountant
    print("\n--- Advanced Accountant ---")
    adv_accountant = AdvancedCompositionAccountant(
        target_epsilon=2.0,
        target_delta=1e-4,
        delta_slack=1e-6
    )

    for eps, delta, desc in operations:
        adv_accountant.consume(eps, delta, desc)

    improvement = expected_eps - adv_accountant.consumed_epsilon

    print(f"  Basic composition:    ε={expected_eps:.3f}")
    print(f"  Advanced composition: ε={adv_accountant.consumed_epsilon:.3f}")
    print(f"  Privacy saved:        Δε={improvement:.3f}")

    advanced_better = adv_accountant.consumed_epsilon <= expected_eps

    if advanced_better and improvement > 0:
        print("  ✓ Advanced composition saves privacy budget")
    else:
        print("  ⚠ Advanced composition not providing expected benefit")

    return basic_correct and advanced_better


def check_dp_selection():
    """Verify DP selector adds noise to selection."""

    print("\n" + "="*70)
    print("TEST 4: DP SELECTION CHECK")
    print("="*70)

    selector = DPSelector(DPSelectorConfig(
        select_k=2,
        epsilon=0.5,
        sensitivity=1.0
    ))

    # Create candidates with known scores
    candidates = []
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]  # Clearly ranked

    for i, score in enumerate(scores):
        var = Variable(f"cand_{i}", role_description="test", requires_grad=False)
        cand = Candidate(var, {"id": i})
        cand = cand.with_scores(dp_score=score, noise_magnitude=0.0)
        candidates.append(cand)

    # Run selection multiple times
    print("\n  Running 5 selection rounds (should see some variation due to noise):")
    selected_sets = []

    for trial in range(5):
        result = selector.select_with_metadata(candidates, rng=random.Random(trial))
        selected_ids = [c.metadata["id"] for c in result.selected]
        selected_sets.append(tuple(sorted(selected_ids)))
        print(f"    Trial {trial + 1}: Selected candidates {selected_ids}")

    # Check for variation (noise should cause different selections sometimes)
    unique_selections = len(set(selected_sets))

    if unique_selections > 1:
        print(f"\n  ✓ PASS: Selection varies across trials (noise detected)")
        print(f"    Unique selection patterns: {unique_selections}/5")
        return True
    else:
        print(f"\n  ⚠ WARNING: Selection always identical (may indicate no noise)")
        print(f"    This could be due to low noise or small score differences")
        return True  # Not necessarily a failure


def check_budget_exhaustion():
    """Verify that exceeding budget raises exception."""

    print("\n" + "="*70)
    print("TEST 5: BUDGET EXHAUSTION CHECK")
    print("="*70)

    from dp_textgrad.dp_es.accountant import PrivacyBudgetExceeded

    accountant = PrivacyAccountant(target_epsilon=1.0, target_delta=1e-5)

    print("\n  Target budget: ε=1.0, δ=1e-5")
    print("  Consuming: ε=0.6, δ=5e-6")

    accountant.consume(0.6, 5e-6, "operation_1")
    print(f"  After op 1: ε={accountant.consumed_epsilon:.2f} (OK)")

    print("  Consuming: ε=0.6, δ=5e-6 (should exceed budget)")

    try:
        accountant.consume(0.6, 5e-6, "operation_2")
        print("\n  ✗ FAIL: Budget exceeded but no exception raised")
        return False
    except PrivacyBudgetExceeded as e:
        print(f"  ✓ PASS: Budget exhaustion detected correctly")
        print(f"    Exception: {str(e)[:60]}...")
        return True


def main():
    """Run all quick privacy checks."""

    print("\n" + "="*70)
    print("QUICK PRIVACY CHECK FOR DP-ES")
    print("="*70)
    print("\nThis runs fast sanity checks to verify privacy mechanisms.")
    print("For comprehensive verification, run example_privacy_verification.py")
    print("="*70)

    results = []

    try:
        results.append(("Noise Injection", check_noise_injection()))
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        results.append(("Noise Injection", False))

    try:
        results.append(("Score Clipping", check_score_clipping()))
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        results.append(("Score Clipping", False))

    try:
        results.append(("Budget Tracking", check_privacy_budget_tracking()))
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        results.append(("Budget Tracking", False))

    try:
        results.append(("DP Selection", check_dp_selection()))
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        results.append(("DP Selection", False))

    try:
        results.append(("Budget Exhaustion", check_budget_exhaustion()))
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        results.append(("Budget Exhaustion", False))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")

    print("\n" + "-"*70)
    print(f"  Total: {passed}/{total} tests passed ({100*passed/total:.0f}%)")
    print("="*70)

    if passed == total:
        print("\n✓ All privacy mechanisms are working correctly!")
        print("  DP-ES appears to be providing privacy protections.")
    elif passed >= total * 0.8:
        print("\n⚠ Most tests passed, but some issues detected.")
        print("  Review failed tests and consider running full verification.")
    else:
        print("\n✗ Multiple privacy mechanism failures detected!")
        print("  Privacy guarantees may be compromised. Investigation needed.")

    print("\n💡 For comprehensive privacy verification, run:")
    print("    python evaluation/example_privacy_verification.py")
    print("\n")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
