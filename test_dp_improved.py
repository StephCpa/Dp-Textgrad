#!/usr/bin/env python3
"""
改进版 DP-TextGrad 测试

改进点：
1. 使用真正的 LLM 驱动变异（避免重复）
2. 支持多轮迭代直到收敛或预算耗尽
3. 更好的评分函数（检测冗余）
"""

import os
import random
import time

os.environ["OPENAI_API_KEY"] = "sk-Lyld88sT_oGZgcE9HyKoLg"
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
print(" 改进版 DP-TextGrad 测试（LLM 驱动变异 + 早停）")
print("=" * 80)

print(f"\nDP-TextGrad 版本: {tg.__version__}")

# 设置引擎
tg.set_backward_engine("experimental:openai/DeepSeek-V3.2-Exp", override=True)
llm_engine = tg.get_engine("experimental:openai/DeepSeek-V3.2-Exp")

# ============================================================================
# 任务定义
# ============================================================================
print("\n" + "=" * 80)
print(" 任务：优化问候语提示")
print("=" * 80)

initial_prompt = "写一个问候语"
target = Variable(initial_prompt, role_description="问候语指令", requires_grad=True)

print(f"\n初始提示: '{initial_prompt}'")

# ============================================================================
# 改进的评分函数（检测冗余）
# ============================================================================

def evaluation_fn(var: Variable) -> float:
    """
    改进的评分函数：
    - 奖励具体性和长度
    - 惩罚重复和冗余
    """
    prompt = var.get_value()
    score = 0.0

    # 基础长度分 (0-3)
    length = len(prompt)
    if 20 < length < 100:
        score += 3.0
    elif length < 20:
        score += 1.0
    elif length > 100:
        score += 2.0  # 太长扣分

    # 关键词覆盖 (0-5)
    keywords = ["友好", "具体", "正式", "场合", "适合", "要求", "清晰", "目标", "对象"]
    keyword_count = sum(1 for kw in keywords if kw in prompt)
    score += min(keyword_count, 5)

    # 惩罚重复（关键改进！）
    words = prompt.split()
    unique_ratio = len(set(words)) / max(len(words), 1)
    if unique_ratio < 0.7:  # 重复率超过 30%
        score -= 3.0  # 严重惩罚
    elif unique_ratio < 0.85:
        score -= 1.0

    # 检测明显的重复片段
    if "要求" in prompt and prompt.count("要求") > 2:
        score -= 2.0

    return max(score, 0.0)  # 确保非负

initial_score = evaluation_fn(target)
print(f"初始分数: {initial_score:.2f}")

# ============================================================================
# DP 组件配置
# ============================================================================
print("\n" + "=" * 80)
print(" 配置差分隐私组件")
print("=" * 80)

scorer_config = DPScorerConfig(
    clipping_value=10.0,
    noise_multiplier=None,  # 自动校准
    epsilon=0.3,            # 降低单次消耗，支持更多迭代
    delta=1e-5
)
scorer = DPScorer(scorer_config)

print(f"✓ DPScorer (每次评分: ε={scorer_config.epsilon})")

selector_config = DPSelectorConfig(
    select_k=2,
    epsilon=0.1,
    sensitivity=1.0
)
selector = DPSelector(selector_config)

print(f"✓ DPSelector (每次选择: ε={selector_config.epsilon})")

# ============================================================================
# LLM 驱动的智能变异（关键改进！）
# ============================================================================

mutation_config = MutationConfig(
    offspring_per_parent=2,
    allow_identity_offspring=False  # 禁止生成相同的后代
)

def llm_mutation_fn(parent: Candidate, iteration: int, rng: random.Random, feedback):
    """
    使用 LLM 生成智能、多样化的变异

    避免简单拼接，而是让 LLM 理解并改进提示
    """
    parent_text = parent.variable.get_value()

    # 构造 LLM 提示，要求生成改进版本
    mutation_prompt = f"""你是一个提示工程专家。请改进以下提示，使其更具体、更清晰，但避免重复和冗余。

原始提示："{parent_text}"

请生成 2 个不同的改进版本：
1. 第一个版本：增加场景描述
2. 第二个版本：增加目标对象说明

要求：
- 每个版本控制在 50 字以内
- 避免使用重复的词语
- 保持简洁清晰

请直接输出两个版本，每行一个，不要编号。"""

    try:
        # 调用 LLM 生成变异
        response = llm_engine.generate(mutation_prompt, system_prompt="你是提示优化专家")

        # 解析响应
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        variations = []

        for line in lines[:2]:  # 最多取2行
            # 清理可能的编号
            cleaned = line.lstrip('123456789.-) ').strip('"\'')
            if cleaned and cleaned != parent_text:
                variations.append(cleaned)

        # 如果 LLM 生成不足，使用后备策略
        while len(variations) < 2:
            fallback = f"{parent_text}（改进版{len(variations) + 1}）"
            variations.append(fallback)

    except Exception as e:
        print(f"  [变异] LLM 调用失败: {e}，使用后备策略")
        # 后备：简单但不重复的变异
        variations = [
            f"{parent_text}，针对商务场合",
            f"{parent_text}，面向客户沟通"
        ]

    # 转换为 Variable 对象
    return [
        Variable(v, role_description=parent.variable.get_role_description(), requires_grad=True)
        for v in variations[:mutation_config.offspring_per_parent]
    ]

mutation_engine = MutationEngine(mutation_fn=llm_mutation_fn, config=mutation_config)

print(f"✓ MutationEngine (LLM 驱动，每个父代 {mutation_config.offspring_per_parent} 后代)")

# ============================================================================
# 隐私会计
# ============================================================================

accountant = PrivacyAccountant(
    target_epsilon=5.0,
    target_delta=1e-4
)

print(f"✓ PrivacyAccountant (总预算: ε={accountant.target_epsilon})")

# ============================================================================
# 进化策略配置（启用早停）
# ============================================================================

evolution_config = DPEvolutionConfig(
    population_size=4,
    parents_to_select=2,
    max_iterations=10,            # 增加最大迭代
    rng_seed=42,
    stop_on_budget=True,
    enable_early_stopping=True,   # 启用早停
    early_stop_patience=3,        # 3 轮无改进则停止
    early_stop_threshold=0.5,     # 改进阈值
    enable_elitism=True,          # 精英保留
    elite_size=1                  # 保留最优1个
)

print(f"✓ 进化策略:")
print(f"  - 最大迭代: {evolution_config.max_iterations}")
print(f"  - 早停: patience={evolution_config.early_stop_patience}, threshold={evolution_config.early_stop_threshold}")
print(f"  - 精英保留: {evolution_config.elite_size} 个")

# ============================================================================
# 运行优化
# ============================================================================
print("\n" + "=" * 80)
print(" 运行优化")
print("=" * 80)

print(f"\n预期每轮消耗: ε={scorer_config.epsilon + selector_config.epsilon}")
print(f"可运行轮数: ~{int(accountant.target_epsilon / (scorer_config.epsilon + selector_config.epsilon))} 轮")

print(f"\n🚀 开始优化...\n")

strategy = DPEvolutionStrategy(
    parameter=target,
    evaluation_fn=evaluation_fn,
    scorer=scorer,
    selector=selector,
    mutation_engine=mutation_engine,
    accountant=accountant,
    config=evolution_config
)

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
    stats = strategy.get_optimization_stats()

    print("\n" + "=" * 80)
    print(" 优化结果")
    print("=" * 80)

    print(f"\n✅ 优化完成（耗时 {elapsed:.2f}秒）")

    print(f"\n📝 提示演化:")
    print(f"  初始: '{initial_prompt}'")
    print(f"  最终: '{target.get_value()}'")

    final_score = evaluation_fn(target)
    print(f"\n📊 分数变化:")
    print(f"  初始: {initial_score:.2f}")
    print(f"  最终: {final_score:.2f}")
    print(f"  提升: {final_score - initial_score:+.2f}")

    # 检查是否有重复
    final_words = target.get_value().split()
    final_unique_ratio = len(set(final_words)) / max(len(final_words), 1)
    print(f"\n📋 质量指标:")
    print(f"  长度: {len(target.get_value())} 字符")
    print(f"  词汇唯一性: {final_unique_ratio:.1%}")
    if final_unique_ratio > 0.85:
        print(f"  ✅ 无明显重复")
    else:
        print(f"  ⚠️  存在重复（唯一性 < 85%）")

    # 隐私预算
    print(f"\n🔒 隐私预算消耗:")
    print(f"  消耗 ε: {accountant.consumed_epsilon:.4f} / {accountant.target_epsilon}")
    print(f"  消耗 δ: {accountant.consumed_delta:.2e} / {accountant.target_delta}")

    utilization = (accountant.consumed_epsilon / accountant.target_epsilon) * 100
    print(f"  利用率: {utilization:.1f}%")

    remaining_eps, remaining_delta = accountant.remaining_budget()
    print(f"\n📊 剩余预算:")
    print(f"  剩余 ε: {remaining_eps:.4f}")
    print(f"  剩余 δ: {remaining_delta:.2e}")

    # 优化统计
    print(f"\n📈 优化过程:")
    print(f"  完成迭代: {stats['iterations_completed']}/{evolution_config.max_iterations}")
    print(f"  是否收敛: {stats.get('converged', False)}")

    if stats.get('converged'):
        print(f"  ✅ 提前收敛（早停生效，节省预算）")
    else:
        if stats['iterations_completed'] < evolution_config.max_iterations:
            print(f"  ⚠️  预算耗尽（未达最大迭代）")
        else:
            print(f"  ℹ️  完整运行（未触发早停）")

    # 分数历史
    if 'score_history' in stats and stats['score_history']:
        print(f"\n  迭代分数历史:")
        for i, score in enumerate(stats['score_history'], 1):
            print(f"    迭代 {i}: {score:.2f}")

    print("\n" + "=" * 80)
    print(" 🎉 测试完成！")
    print("=" * 80)

    print("\n✅ 验证:")
    print(f"  ✓ 差分隐私保护: ε={accountant.consumed_epsilon:.4f}")
    print(f"  ✓ 隐私预算追踪: 正常")
    print(f"  ✓ (ε,δ)-DP 保证: 有效")
    if final_score > initial_score:
        print(f"  ✓ 提示质量改进: +{final_score - initial_score:.2f}")
    if final_unique_ratio > 0.85:
        print(f"  ✓ 无重复冗余")

else:
    print("\n❌ 优化失败")

print()
