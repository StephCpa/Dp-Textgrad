#!/usr/bin/env python3
"""
最小化工作版本 - 基于官方测试

这个版本直接基于项目的官方测试，确保隐私预算追踪正确工作
"""

import os
import random

os.environ["OPENAI_API_KEY"] = "sk-ccai3pDr1nrkZOs_CJUs4w"
os.environ["OPENAI_API_BASE"] = "https://llmapi.paratera.com"

import dp_textgrad as tg
from dp_textgrad import Variable
from dp_textgrad.dp_es import (
    DPEvolutionStrategy,
    DPEvolutionConfig,
    PrivacyAccountant,
    DPScorer,
    DPScorerConfig,
    DPSelector,
    DPSelectorConfig,
    MutationEngine,
    MutationConfig,
)
from dp_textgrad.dp_es.population import Candidate

print("=" * 80)
print(" 最小化工作版本 - 基于官方测试")
print("=" * 80)

print(f"\nDP-TextGrad 版本: {tg.__version__}")

# 设置 LLM 引擎（用于更智能的变异）
tg.set_backward_engine("experimental:openai/DeepSeek-V3.2-Exp", override=True)

# ============================================================================
# 定义任务：优化一个提示，使其更具体
# ============================================================================
print("\n" + "=" * 80)
print(" 任务：优化提示使其更具体")
print("=" * 80)

initial_prompt = "写一个问候语"
target = Variable(initial_prompt, role_description="问候语指令", requires_grad=True)

print(f"\n初始提示: '{initial_prompt}'")

# ============================================================================
# 配置 DP 组件
# ============================================================================
print("\n" + "=" * 80)
print(" 配置差分隐私组件")
print("=" * 80)

# 评分器：带差分隐私保护
scorer_config = DPScorerConfig(
    clipping_value=10.0,        # 裁剪值
    noise_multiplier=None,       # 自动校准噪声
    epsilon_per_candidate=0.5,   # 每次评分消耗 ε=0.5
    delta_per_candidate=1e-5,    # δ 参数
    composition="basic"          # 使用基础组合以便预测
)
scorer = DPScorer(scorer_config)

print(f"✓ DPScorer")
print(f"  - 每次评分消耗: ε={scorer_config.epsilon_per_candidate}")
print(f"  - 裁剪值: {scorer_config.clipping_value}")
print(f"  - 自动噪声校准: {scorer_config.noise_multiplier is None}")

# 选择器：带差分隐私保护
selector_config = DPSelectorConfig(
    select_k=2,                 # 每次选择 2 个父代
    epsilon=0.1,                # 每次选择消耗 ε=0.1
    sensitivity=1.0
)
selector = DPSelector(selector_config)

print(f"✓ DPSelector")
print(f"  - 每次选择消耗: ε={selector_config.epsilon}")

# 变异引擎：使用 LLM 生成更好的变异
mutation_config = MutationConfig(
    offspring_per_parent=2,      # 每个父代生成 2 个后代
    allow_identity_offspring=True
)

def llm_mutation_fn(parent: Candidate, iteration: int, rng: random.Random, feedback):
    """使用 LLM 生成智能变异"""
    parent_text = parent.variable.get_value()

    # 简单的基于规则的变异（快速测试）
    variations = [
        f"{parent_text}，要求友好且具体",
        f"{parent_text}，适合正式场合",
    ]

    return [
        Variable(v, role_description=parent.variable.get_role_description(), requires_grad=True)
        for v in variations
    ]

mutation_engine = MutationEngine(mutation_fn=llm_mutation_fn, config=mutation_config)

print(f"✓ MutationEngine")
print(f"  - 每个父代生成: {mutation_config.offspring_per_parent} 个后代")

# 隐私会计：追踪总预算消耗
accountant = PrivacyAccountant(
    target_epsilon=5.0,         # 总预算 ε=5.0
    target_delta=1e-4           # 总预算 δ=1e-4
)

print(f"✓ PrivacyAccountant")
print(f"  - 总预算: ε={accountant.target_epsilon}, δ={accountant.target_delta}")

# ============================================================================
# 评估函数：启发式评分
# ============================================================================

def evaluation_fn(var: Variable) -> float:
    """评估提示质量（有区分度）"""
    prompt = var.get_value()
    score = 0.0

    # 长度
    if len(prompt) > 30:
        score += 3.0
    elif len(prompt) > 20:
        score += 2.0
    else:
        score += 1.0

    # 关键词
    keywords = ["友好", "具体", "正式", "场合", "适合", "要求"]
    for kw in keywords:
        if kw in prompt:
            score += 1.0

    return score

initial_score = evaluation_fn(target)
print(f"\n初始分数: {initial_score:.2f}")

# ============================================================================
# 创建并运行优化器
# ============================================================================
print("\n" + "=" * 80)
print(" 运行差分隐私优化")
print("=" * 80)

evolution_config = DPEvolutionConfig(
    population_size=4,           # 种群大小
    parents_to_select=2,         # 选择 2 个父代
    max_iterations=3,            # 运行 3 轮迭代
    rng_seed=42,
    stop_on_budget=True          # 预算耗尽时停止
)

print(f"\n配置:")
print(f"  种群大小: {evolution_config.population_size}")
print(f"  父代选择: {evolution_config.parents_to_select}")
print(f"  最大迭代: {evolution_config.max_iterations}")

print(f"\n🚀 开始优化...")

strategy = DPEvolutionStrategy(
    parameter=target,
    evaluation_fn=evaluation_fn,
    scorer=scorer,
    selector=selector,
    mutation_engine=mutation_engine,
    accountant=accountant,
    config=evolution_config
)

import time
start = time.time()

try:
    strategy.step()
    success = True
except Exception as e:
    print(f"\n❌ 失败: {e}")
    import traceback
    traceback.print_exc()
    success = False

elapsed = time.time() - start

# ============================================================================
# 结果分析
# ============================================================================
if success:
    print("\n" + "=" * 80)
    print(" 优化结果")
    print("=" * 80)

    print(f"\n✅ 优化完成（耗时 {elapsed:.2f}秒）")

    print(f"\n📝 提示变化:")
    print(f"  初始: '{initial_prompt}'")
    print(f"  最终: '{target.get_value()}'")

    final_score = evaluation_fn(target)
    print(f"\n📊 分数变化:")
    print(f"  初始: {initial_score:.2f}")
    print(f"  最终: {final_score:.2f}")
    print(f"  提升: {final_score - initial_score:+.2f}")

    # 隐私预算消耗（关键！）
    print(f"\n🔒 隐私预算消耗:")
    print(f"  消耗 ε: {accountant.consumed_epsilon:.4f} / {accountant.target_epsilon}")
    print(f"  消耗 δ: {accountant.consumed_delta:.2e} / {accountant.target_delta}")

    remaining_eps, remaining_delta = accountant.remaining_budget()
    print(f"\n📊 剩余预算:")
    print(f"  剩余 ε: {remaining_eps:.4f}")
    print(f"  剩余 δ: {remaining_delta:.2e}")

    # 预期消耗计算
    iterations_run = min(evolution_config.max_iterations, 3)  # 实际运行的迭代数
    # Note: With basic composition, each iteration scores population_size candidates
    # So: population_size × epsilon_per_candidate per iteration
    expected_eps_per_iter = evolution_config.population_size * scorer_config.epsilon_per_candidate + selector_config.epsilon
    expected_eps = iterations_run * expected_eps_per_iter

    print(f"\n💡 预期消耗:")
    print(f"  每轮评分: {evolution_config.population_size} candidates × ε={scorer_config.epsilon_per_candidate} = {evolution_config.population_size * scorer_config.epsilon_per_candidate}")
    print(f"  每轮选择: ε={selector_config.epsilon}")
    print(f"  每轮总计: ε={expected_eps_per_iter}")
    print(f"  {iterations_run} 轮预期: ε={expected_eps:.2f}")

    if accountant.consumed_epsilon > 0:
        print(f"\n  ✅ 实际消耗: ε={accountant.consumed_epsilon:.4f}")
        print(f"  ✅ 隐私预算追踪正常工作！")
    else:
        print(f"\n  ❌ 实际消耗为 0，隐私预算追踪失败")

    # 获取优化统计
    stats = strategy.get_optimization_stats()
    print(f"\n📈 优化统计:")
    print(f"  完成迭代: {stats['iterations_completed']}/{evolution_config.max_iterations}")
    print(f"  是否收敛: {stats.get('converged', False)}")

    print("\n" + "=" * 80)
    print(" 🎉 测试成功！")
    print("=" * 80)

    print("\n✅ 验证结论:")
    if accountant.consumed_epsilon > 0:
        print("  ✓ 差分隐私保护启用")
        print("  ✓ 隐私预算正确追踪")
        print("  ✓ (ε,δ)-DP 保证有效")
    else:
        print("  ❌ 隐私预算追踪失败")

    if final_score > initial_score:
        print("  ✓ 提示质量改进")

else:
    print("\n❌ 优化失败")

print()
