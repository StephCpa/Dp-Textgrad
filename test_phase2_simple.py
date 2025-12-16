"""简化的 Phase 2 验证测试"""

import warnings
from dp_textgrad.dp_es import DPScorer, DPScorerConfig
from dp_textgrad.dp_es.population import Candidate
from dp_textgrad import Variable

print("="*80)
print("Phase 2 隐私修复简化验证")
print("="*80)

# 测试 1: 基础组合计算
print("\n测试 1: 基础组合隐私预算计算")
print("-"*80)

config = DPScorerConfig(
    epsilon_per_candidate=0.5,
    delta_per_candidate=1e-5,
    composition="basic",
    clipping_value=1.0,
)

scorer = DPScorer(config)

candidates = [
    Candidate(variable=Variable(f"test {i}", role_description="test"))
    for i in range(4)
]

scores = scorer.evaluate(candidates, lambda c: 1.0)

print(f"候选数量: 4")
print(f"每个候选: ε={config.epsilon_per_candidate}, δ={config.delta_per_candidate:.0e}")
print(f"组合方法: {config.composition}")
print(f"总预算: ε={scores.epsilon:.4f}, δ={scores.delta:.2e}")

# 验证: 基础组合应该是 4 × 0.5 = 2.0
assert abs(scores.epsilon - 2.0) < 0.01, f"ε 应为 2.0,实际 {scores.epsilon}"
assert abs(scores.delta - 4e-5) < 1e-6, f"δ 应为 4e-5,实际 {scores.delta}"
print("✅ 基础组合计算正确!")

# 测试 2: 反馈默认禁用
print("\n测试 2: 反馈默认禁用")
print("-"*80)

config2 = DPScorerConfig(
    epsilon_per_candidate=0.5,
    enable_feedback=False,  # 默认
)

scorer2 = DPScorer(config2)
candidate = Candidate(variable=Variable("test", role_description="test"))

scores2 = scorer2.evaluate([candidate], lambda c: (1.0, "sensitive feedback"))

assert "dp_feedback" not in scores2.updated_candidates[0].metadata, \
    "反馈应该被禁用"
print("✅ 反馈默认禁用!")

# 测试 3: 启用反馈显示警告
print("\n测试 3: 启用反馈显示警告")
print("-"*80)

config3 = DPScorerConfig(
    epsilon_per_candidate=0.5,
    enable_feedback=True,
)

scorer3 = DPScorer(config3)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    scores3 = scorer3.evaluate([candidate], lambda c: (1.0, "feedback"))
    
    assert len(w) == 1, f"应有 1 个警告,实际 {len(w)} 个"
    assert "NOT differentially private" in str(w[0].message)

print("✅ 警告正确显示!")

# 测试 4: 向后兼容
print("\n测试 4: 向后兼容性")
print("-"*80)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    
    config4 = DPScorerConfig(epsilon_per_candidate=0.5, delta_per_candidate=1e-5)
    
    # 访问旧属性应触发警告
    eps = config4.epsilon
    delta = config4.delta
    
    deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(deprecation_warnings) == 2, \
        f"应有 2 个 DeprecationWarning,实际 {len(deprecation_warnings)} 个"

print("✅ 向后兼容性正常!")

print("\n" + "="*80)
print("🎉 所有测试通过!")
print("="*80)
print("\nPhase 2 修复已验证:")
print("  ✅ 隐私预算正确计算 (基础组合)")
print("  ✅ 反馈默认禁用")
print("  ✅ 启用反馈显示警告")
print("  ✅ 向后兼容性保持")
print("="*80)
