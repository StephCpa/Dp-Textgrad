# Privacy Verification for DP-ES

This directory contains tools to empirically verify that DP-ES provides the promised differential privacy guarantees.

## 📋 Overview

Differential privacy is verified through four types of tests:

### 1. **Neighboring Database Test**
- **What it tests**: Core DP property - outputs on neighboring datasets (differing by 1 record) should be similar
- **How**: Runs optimization on D and D' (with one record removed), compares outputs
- **Pass criteria**: Output differences consistent with (ε, δ) bounds

### 2. **Membership Inference Attack**
- **What it tests**: Resistance to determining if a specific record was in the training data
- **How**: Simulates attacker trying to infer membership from outputs
- **Pass criteria**: Attack success rate ≤ random guessing + theoretical advantage

### 3. **Noise Distribution Verification**
- **What it tests**: Injected noise follows expected statistical distribution
- **How**: Analyzes noise samples, compares to theoretical Gaussian/Laplace distribution
- **Pass criteria**: KS-test p-value > 0.05, mean ≈ 0, std ≈ expected σ

### 4. **Privacy Budget Tracking**
- **What it tests**: Accountant correctly tracks privacy consumption
- **How**: Simulates multiple operations, verifies cumulative ε and δ
- **Pass criteria**: Reported budget ≤ theoretical basic composition bound

---

## 🚀 Quick Start

### Step 1: Run Privacy Verification Tests

```bash
cd evaluation
python example_privacy_verification.py
```

**Output:**
- Console: Detailed test progress and results
- `privacy_verification_results/verification_report.txt` - Human-readable summary
- `privacy_verification_results/verification_results.json` - Machine-readable data

**Expected runtime**: ~5-10 minutes (depends on number of trials)

### Step 2: Generate Visualizations

```bash
python privacy_visualization.py
```

**Output:**
- `privacy_verification_results/privacy_verification_summary.png` - Overview dashboard
- `privacy_verification_results/privacy_verification_detailed.png` - Per-test breakdown

**Requirements**: `matplotlib` (install with `pip install matplotlib`)

---

## 📊 Understanding Results

### Test Status Indicators

| Status | Meaning |
|--------|---------|
| ✓ PASS | Test confirms privacy guarantees hold |
| ✗ FAIL | Test suggests potential privacy violation |
| ⚠️ WARNING | Borderline case, review metrics carefully |

### Key Metrics

#### Neighboring Database Test
- `mean_difference`: Average output change when removing one record (should be small)
- `max_privacy_violation`: Worst-case violation (should be ≤ ε)
- `identical_outputs_pct`: % of trials with identical outputs (higher = better)

#### Membership Inference Attack
- `attack_success_rate`: Attacker's success at inferring membership
- `theoretical_advantage`: Maximum advantage allowed by (ε, δ)-DP
- `advantage_over_random`: Actual advantage over 50% random guessing

**Interpretation:**
- Success rate ≤ 55%: ✓ Strong privacy
- Success rate 55-65%: ⚠️ Moderate privacy
- Success rate > 65%: ✗ Weak privacy

#### Noise Distribution
- `observed_std` vs `expected_std`: Should match closely (<20% error)
- `ks_pvalue`: KS-test p-value (>0.05 = distribution matches)
- `std_error`: Relative error in standard deviation

#### Privacy Budget Tracking
- `epsilon_improvement`: Savings from advanced composition
- `consumed_epsilon`: Actual privacy spent
- `expected_epsilon_basic`: What basic composition would spend

---

## 🔬 Advanced Usage

### Custom Privacy Parameters

Edit `example_privacy_verification.py` to test different privacy budgets:

```python
# In main() function, change:
EPSILON = 1.0  # Higher = less privacy, faster convergence
DELTA = 1e-6   # Lower = stricter privacy
```

### Adjusting Test Rigor

Increase trials for more reliable statistical tests:

```python
# In test functions:
test = NeighboringDatabaseTest(
    epsilon=EPSILON,
    delta=DELTA,
    num_trials=100  # Default: 10, increase for better confidence
)
```

### Testing Different Configurations

```python
from dp_textgrad.config_dp import get_dp_config

# Test high privacy configuration
config = get_dp_config("high_privacy")

# Use in optimizer_factory
scorer = DPScorer(config.scorer)
selector = DPSelector(config.selector)
# ... rest of setup
```

---

## 📈 Interpreting Visual Reports

### Summary Dashboard (`privacy_verification_summary.png`)

**Panel 1: Test Results Overview**
- Green bars = tests passed
- Red bars = tests failed

**Panel 2: Privacy Budget per Test**
- Shows ε used by each test
- Lower is better (less privacy cost)

**Panel 3: Membership Inference Resistance**
- Compares observed attack rate vs random guessing
- Closer to 50% = better privacy

**Panel 4: Noise Distribution Validation**
- Expected vs observed noise standard deviation
- Should be nearly identical

### Detailed Report (`privacy_verification_detailed.png`)

Each test gets 3 panels:
1. **Status & Metrics**: Pass/fail status and key numbers
2. **Visualization**: Test-specific chart
3. **Privacy Budget**: ε and δ used

---

## 🛠️ Troubleshooting

### "No module named scipy"

**Solution**: Install scipy for statistical tests
```bash
pip install scipy
```

**Note**: Tests will still run without scipy, but with limited statistical validation.

### "Results file not found"

**Solution**: Run the test script first:
```bash
python example_privacy_verification.py
```

### Tests are failing

**Check:**
1. Privacy parameters: Lower ε (more privacy) may cause tests to fail due to high noise
2. Number of trials: Increase `num_trials` for more stable results
3. Dataset size: Very small datasets (<10 records) may show high variance

**Typical issues:**
- **Membership inference fails**: Attack success rate too high → Increase privacy budget or check noise calibration
- **Noise distribution fails**: σ mismatch → Verify DPScorer configuration
- **Neighboring DB fails**: Too much output variation → Check clipping bounds

---

## 🔐 Privacy Guarantee Interpretation

### What does (ε, δ)-DP mean?

For neighboring datasets D and D' (differing by 1 record):

```
Pr[M(D) ∈ S] ≤ e^ε × Pr[M(D') ∈ S] + δ
```

**Where:**
- `M` = DP-ES optimization mechanism
- `ε` (epsilon) = Privacy loss parameter (smaller = better privacy)
- `δ` (delta) = Probability of privacy violation (should be ≪ 1/n)

### Recommended Values

| Use Case | ε | δ | Notes |
|----------|---|---|-------|
| Healthcare/Finance | ≤ 1.0 | ≤ 10^-6 | Strong privacy |
| General Business | 1.0-3.0 | ≤ 10^-5 | Balanced |
| Research/Aggregated | 3.0-6.0 | ≤ 10^-4 | Moderate privacy |
| Testing | > 6.0 | ≤ 10^-3 | Weak privacy |

---

## 📚 Further Reading

- **Differential Privacy Theory**: [Programming Differential Privacy](https://programming-dp.com/)
- **DP-ES Architecture**: `../DP-TextGrad via DP-ES.md`
- **Membership Inference Attacks**: Shokri et al., "Membership Inference Attacks Against Machine Learning Models"
- **Composition Theorems**: Dwork & Roth, "The Algorithmic Foundations of Differential Privacy"

---

## 🤝 Contributing

To add new privacy tests:

1. Create a new test class in `privacy_verification.py`
2. Implement `run_test()` method returning `PrivacyTestResult`
3. Add test to `example_privacy_verification.py`
4. Update visualization in `privacy_visualization.py`

---

## 📝 Citation

If you use this privacy verification framework, please cite:

```bibtex
@article{textgrad2024,
  title={TextGrad: Automatic Differentiation via Text},
  journal={Nature},
  year={2025}
}
```

---

## 📞 Support

- **Issues**: Report bugs at [GitHub Issues](https://github.com/zou-group/dp-textgrad/issues)
- **Questions**: See main project README or documentation
- **Privacy concerns**: Review `DP-TextGrad via DP-ES.md` design document
