"""
Test to verify DPScoreRecord leakage fix

Verifies that:
1. By default (unsafe_debug_mode=False), records are empty
2. With unsafe_debug_mode=True, records are populated with warning
3. Production code cannot access raw scores without explicit opt-in
"""

import warnings
from dp_textgrad.dp_es import DPScorer, DPScorerConfig
from dp_textgrad.dp_es.population import Candidate
from dp_textgrad import Variable


def test_production_mode_protects_privacy():
    """Test 1: Default mode (production) - records should be empty"""
    print("\n" + "="*80)
    print("TEST 1: Production Mode (unsafe_debug_mode=False, default)")
    print("="*80)

    config = DPScorerConfig(
        epsilon_per_candidate=1.0,
        delta_per_candidate=1e-5,
        clipping_value=1.0,
        noise_multiplier=1.0,
        # unsafe_debug_mode=False  # Default!
    )
    scorer = DPScorer(config)

    candidates = [
        Candidate(Variable(f"test_{i}", role_description="test"))
        for i in range(3)
    ]

    def eval_fn(cand):
        return 0.5

    scores = scorer.evaluate(candidates, eval_fn)

    # CRITICAL: In production mode, records should be EMPTY
    print(f"\nNumber of candidates: {len(candidates)}")
    print(f"Number of records: {len(scores.records)}")
    print(f"Records list: {scores.records}")

    if len(scores.records) == 0:
        print("\n✅ PASS: Records are empty in production mode")
        print("   Raw scores are NOT accessible - privacy protected!")
        return True
    else:
        print("\n❌ FAIL: Records are populated in production mode")
        print("   This is a CRITICAL privacy leak!")
        return False


def test_debug_mode_exposes_data_with_warning():
    """Test 2: Debug mode - records populated but with clear warning"""
    print("\n" + "="*80)
    print("TEST 2: Debug Mode (unsafe_debug_mode=True)")
    print("="*80)

    config = DPScorerConfig(
        epsilon_per_candidate=1.0,
        delta_per_candidate=1e-5,
        clipping_value=1.0,
        noise_multiplier=0.0,  # No noise for clear test
        unsafe_debug_mode=True,  # EXPLICITLY enabled
    )
    scorer = DPScorer(config)

    candidates = [
        Candidate(Variable(f"test_{i}", role_description="test"))
        for i in range(3)
    ]

    def eval_fn(cand):
        return 0.5

    # This should trigger the warning
    print("\nEvaluating candidates (should show privacy warning)...")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        scores = scorer.evaluate(candidates, eval_fn)

        # Check if warning was shown
        warning_shown = any("unsafe_debug_mode" in str(warning.message) for warning in w)

    print(f"\nNumber of records: {len(scores.records)}")
    print(f"Warning shown: {warning_shown}")

    if len(scores.records) > 0:
        record = scores.records[0]
        print(f"\nRecord 0 contents:")
        print(f"  raw_score: {record.raw_score}")
        print(f"  clipped_score: {record.clipped_score}")
        print(f"  noise: {record.noise}")
        print(f"  dp_score: {record.dp_score}")

    if len(scores.records) == 3 and warning_shown:
        print("\n✅ PASS: Debug mode works correctly")
        print("   - Records are populated")
        print("   - Privacy warning was shown")
        return True
    else:
        print("\n❌ FAIL: Debug mode not working correctly")
        return False


def test_production_code_cannot_leak():
    """Test 3: Simulate production code trying to access records"""
    print("\n" + "="*80)
    print("TEST 3: Production Code Cannot Access Raw Scores")
    print("="*80)

    # Typical production configuration
    config = DPScorerConfig(
        epsilon_per_candidate=0.5,
        delta_per_candidate=1e-5,
    )
    scorer = DPScorer(config)

    candidates = [
        Candidate(Variable(f"candidate_{i}", role_description="solution"))
        for i in range(5)
    ]

    def eval_fn(cand):
        # Simulate evaluation with different scores
        idx = int(cand.variable.get_value().split("_")[1])
        return 0.1 * idx

    scores = scorer.evaluate(candidates, eval_fn)

    # Try to access raw scores like the old vulnerable code did
    print("\nAttempting to access raw scores (like vulnerable code)...")
    try:
        if len(scores.records) > 0:
            raw_score = scores.records[0].raw_score
            print(f"❌ LEAKED: raw_score = {raw_score}")
            print("   This is a privacy breach!")
            return False
        else:
            print("✅ PROTECTED: records list is empty")
            print("   Cannot access raw scores")
            print(f"   scores.records = {scores.records}")

            # Verify DP scores are still accessible
            print(f"\n   DP-protected scores ARE available:")
            for i, cand in enumerate(scores.updated_candidates):
                print(f"     Candidate {i}: dp_score = {cand.dp_score:.3f}")

            return True
    except IndexError:
        print("✅ PROTECTED: IndexError when accessing records (empty list)")
        return True


def test_updated_candidates_still_work():
    """Test 4: Verify normal operations still work without records"""
    print("\n" + "="*80)
    print("TEST 4: Normal DP Operations Still Work")
    print("="*80)

    config = DPScorerConfig(
        epsilon_per_candidate=1.0,
        delta_per_candidate=1e-5,
        clipping_value=2.0,
        noise_multiplier=1.0,
    )
    scorer = DPScorer(config)

    candidates = [
        Candidate(Variable(f"solution_{i}", role_description="test"))
        for i in range(4)
    ]

    scores_list = [0.5, 1.0, 1.5, 2.0]

    def eval_fn(cand):
        idx = int(cand.variable.get_value().split("_")[1])
        return scores_list[idx]

    scores = scorer.evaluate(candidates, eval_fn)

    # Verify all normal operations work
    checks_passed = []

    # Check 1: Privacy budget is calculated
    checks_passed.append(("Privacy budget calculated", scores.epsilon > 0))

    # Check 2: Candidates were updated with DP scores
    checks_passed.append(("Candidates updated", len(scores.updated_candidates) == 4))

    # Check 3: DP scores are present
    all_have_scores = all(c.dp_score is not None for c in scores.updated_candidates)
    checks_passed.append(("All have dp_score", all_have_scores))

    # Check 4: Noise magnitude stored (not raw noise!)
    all_have_noise_mag = all(c.noise_magnitude is not None for c in scores.updated_candidates)
    checks_passed.append(("All have noise_magnitude", all_have_noise_mag))

    # Check 5: Metadata is present (adaptive clipping info)
    all_have_metadata = all("dp_adaptive_clipping_value" in c.metadata for c in scores.updated_candidates)
    checks_passed.append(("All have metadata", all_have_metadata))

    print("\nNormal operation checks:")
    for check_name, passed in checks_passed:
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}: {passed}")

    all_passed = all(p for _, p in checks_passed)

    if all_passed:
        print("\n✅ PASS: All normal DP operations work correctly")
        print("   Privacy is protected, functionality is intact!")
        return True
    else:
        print("\n❌ FAIL: Some operations failed")
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("DP-TEXTGRAD: DPScoreRecord LEAKAGE FIX VERIFICATION")
    print("="*80)
    print("\nThis test verifies the fix for the critical privacy leak where")
    print("DPScores.records exposed raw_score, clipped_score, and noise.")
    print("\nAfter the fix:")
    print("  - Production mode (default): records are EMPTY")
    print("  - Debug mode (explicit): records populated WITH WARNING")

    results = []

    results.append(("Production mode protects privacy", test_production_mode_protects_privacy()))
    results.append(("Debug mode works with warning", test_debug_mode_exposes_data_with_warning()))
    results.append(("Production code cannot leak", test_production_code_cannot_leak()))
    results.append(("Normal operations still work", test_updated_candidates_still_work()))

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(p for _, p in results)

    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        print("\nThe DPScoreRecord leakage has been successfully fixed:")
        print("  ✅ Raw scores are NOT accessible in production mode")
        print("  ✅ Debug mode requires explicit opt-in with clear warnings")
        print("  ✅ Normal DP operations continue to work correctly")
        print("\nDifferential privacy protections are now enforced by default!")
    else:
        print("❌ SOME TESTS FAILED")
        print("="*80)
        print("\nThe fix may not be complete. Review the failures above.")

    print("="*80)
