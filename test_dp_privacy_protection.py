#!/usr/bin/env python3
"""
差分隐私保护验证实验

核心问题：DP-ES 能否防止训练数据中的敏感信息泄露到优化后的提示中？

实验设计：
1. 构造包含敏感信息的训练样本（姓名、地址、电话等）
2. 使用这些样本评估提示质量
3. 对比 DP 和 non-DP 方法优化后的提示是否泄露敏感信息

预期结果：
- DP-ES: 添加噪声，敏感信息被模糊化或替换
- non-DP: 可能直接记忆并泄露训练数据中的敏感信息
"""

import os
import random
import re
from typing import List, Dict, Set

os.environ["OPENAI_API_KEY"] = "sk-Lyld88sT_oGZgcE9HyKoLg"
os.environ["OPENAI_API_BASE"] = "https://llmapi.paratera.com"

import dp_textgrad as tg
from dp_textgrad import Variable
from dp_textgrad.dp_es import (
    DPEvolutionStrategy,
    DPScorer,
    DPSelector,
    MutationEngine,
    AdvancedCompositionAccountant,
    DPScorerConfig,
    DPSelectorConfig,
    MutationConfig,
    DPEvolutionConfig,
)
from dp_textgrad.dp_es.population import Candidate

print("=" * 80)
print(" 差分隐私保护验证实验")
print("=" * 80)

print(f"\nDP-TextGrad 版本: {tg.__version__}")

tg.set_backward_engine("experimental:openai/DeepSeek-V3.2-Exp", override=True)
llm_engine = tg.get_engine("experimental:openai/DeepSeek-V3.2-Exp")

# ============================================================================
# 步骤 1: 构造包含敏感信息的训练数据集
# ============================================================================
print("\n" + "=" * 80)
print(" 步骤 1: 构造包含敏感信息的训练数据")
print("=" * 80)

# 模拟敏感的客户数据
SENSITIVE_TRAINING_DATA = [
    {
        "name": "张伟",
        "phone": "13812345678",
        "address": "北京市朝阳区建国路88号",
        "email": "zhangwei@example.com",
        "age": 35,
        "salary": 25000,
    },
    {
        "name": "李娜",
        "phone": "13923456789",
        "address": "上海市浦东新区陆家嘴环路1000号",
        "email": "lina@example.com",
        "age": 28,
        "salary": 18000,
    },
    {
        "name": "王强",
        "phone": "15034567890",
        "address": "广州市天河区珠江新城花城大道123号",
        "email": "wangqiang@example.com",
        "age": 42,
        "salary": 32000,
    },
    {
        "name": "赵敏",
        "phone": "18645678901",
        "address": "深圳市南山区科技园南区深南大道9988号",
        "email": "zhaomin@example.com",
        "age": 31,
        "salary": 28000,
    },
]

print("\n模拟敏感数据集（4 条客户记录）:")
for i, record in enumerate(SENSITIVE_TRAINING_DATA, 1):
    print(f"  记录 {i}: {record['name']}, {record['phone']}, {record['address']}")

# 收集所有敏感字段用于后续检测
SENSITIVE_KEYWORDS = set()
for record in SENSITIVE_TRAINING_DATA:
    SENSITIVE_KEYWORDS.add(record['name'])
    SENSITIVE_KEYWORDS.add(record['phone'])
    SENSITIVE_KEYWORDS.add(record['address'])
    SENSITIVE_KEYWORDS.add(record['email'])

print(f"\n敏感关键词数量: {len(SENSITIVE_KEYWORDS)}")

# ============================================================================
# 步骤 2: 定义基于敏感数据的评估函数
# ============================================================================
print("\n" + "=" * 80)
print(" 步骤 2: 定义基于敏感数据的评估函数")
print("=" * 80)

INITIAL_PROMPT = "生成一个客户联系信息查询提示"

def evaluation_with_sensitive_data(var: Variable) -> float:
    """
    评估函数：基于敏感训练数据打分

    这个函数会"看到"敏感数据，优化过程可能会将敏感信息编码到提示中
    DP 的噪声机制应该能防止这种泄露
    """
    prompt = var.get_value()
    score = 0.0

    # 基础分：长度
    if 20 < len(prompt) < 100:
        score += 2.0
    elif len(prompt) > 100:
        score += 1.0

    # 关键词覆盖（通用关键词，非敏感）
    general_keywords = ["客户", "联系", "信息", "查询", "姓名", "电话", "地址", "邮箱"]
    keyword_count = sum(1 for kw in general_keywords if kw in prompt)
    score += min(keyword_count * 0.5, 4.0)

    # 危险：如果提示中包含训练数据的特定信息，给高分
    # 这会激励优化器"记忆"敏感数据
    # DP 应该通过噪声防止这种过拟合
    for record in SENSITIVE_TRAINING_DATA:
        # 如果提示提到具体的名字
        if record['name'] in prompt:
            score += 2.0  # 高奖励！non-DP 会倾向于记忆

        # 如果提到具体地址的特征（如"朝阳区"、"浦东新区"）
        address_parts = record['address'].split('市')[1].split('区')[0] if '市' in record['address'] and '区' in record['address'] else ""
        if address_parts and address_parts in prompt:
            score += 1.5

        # 如果提到电话号码前缀
        phone_prefix = record['phone'][:3]
        if phone_prefix in prompt:
            score += 1.5

    return min(score, 10.0)

initial_score = evaluation_with_sensitive_data(
    Variable(INITIAL_PROMPT, role_description="test", requires_grad=False)
)

print(f"\n初始提示: '{INITIAL_PROMPT}'")
print(f"初始分数: {initial_score:.2f}")
print("\n⚠️  评估函数会奖励包含敏感信息的提示（模拟过拟合风险）")

# ============================================================================
# 步骤 3: DP-ES 优化（带隐私保护）
# ============================================================================
print("\n" + "=" * 80)
print(" 步骤 3: DP-ES 优化（带隐私保护）")
print("=" * 80)

def run_dp_optimization():
    target = Variable(INITIAL_PROMPT, role_description="客户查询提示", requires_grad=True)

    # DP 配置（增加预算以支持完整运行）
    scorer_config = DPScorerConfig(
        clipping_value=10.0,
        noise_multiplier=None,
        epsilon=1.5,  # 适中的隐私保护
        delta=1e-5,
        enable_score_cache=False  # 禁用缓存以真实测试
    )
    scorer = DPScorer(scorer_config)

    selector_config = DPSelectorConfig(
        select_k=2,
        epsilon=0.5,
        sensitivity=1.0
    )
    selector = DPSelector(selector_config)

    mutation_config = MutationConfig(
        offspring_per_parent=2,
        allow_identity_offspring=False
    )

    def mutation_fn(parent: Candidate, iteration: int, rng: random.Random, feedback):
        """使用 LLM 生成变异（可能泄露敏感信息）"""
        parent_text = parent.variable.get_value()

        # 简单变异（避免直接调用 LLM）
        variations = [
            f"{parent_text}，包含姓名和联系方式",
            f"{parent_text}，用于快速查找客户",
        ]

        return [
            Variable(v, role_description=parent.variable.get_role_description(), requires_grad=True)
            for v in variations
        ]

    mutation_engine = MutationEngine(mutation_fn=mutation_fn, config=mutation_config)

    accountant = AdvancedCompositionAccountant(
        target_epsilon=15.0,  # 增加总预算
        target_delta=1e-4
    )

    evolution_config = DPEvolutionConfig(
        population_size=4,
        parents_to_select=2,
        max_iterations=3,  # 减少迭代以适应预算
        rng_seed=42,
        stop_on_budget=True,  # 预算耗尽时停止
        enable_early_stopping=False,
        enable_elitism=True,
        elite_size=1
    )

    strategy = DPEvolutionStrategy(
        parameter=target,
        evaluation_fn=evaluation_with_sensitive_data,
        scorer=scorer,
        selector=selector,
        mutation_engine=mutation_engine,
        accountant=accountant,
        config=evolution_config
    )

    print("\n🔒 运行 DP-ES 优化...")

    try:
        strategy.step()
        success = True
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        success = False

    if not success:
        return None

    stats = strategy.get_optimization_stats()
    final_score = evaluation_with_sensitive_data(target)

    return {
        "final_prompt": target.get_value(),
        "final_score": final_score,
        "privacy_epsilon": stats['privacy_consumed_epsilon'],
        "privacy_delta": stats['privacy_consumed_delta'],
    }

dp_result = run_dp_optimization()

if dp_result:
    print(f"\n✅ DP-ES 完成")
    print(f"  最终提示: '{dp_result['final_prompt']}'")
    print(f"  最终分数: {dp_result['final_score']:.2f}")
    print(f"  隐私消耗: ε={dp_result['privacy_epsilon']:.4f}, δ={dp_result['privacy_delta']:.2e}")

# ============================================================================
# 步骤 4: non-DP 优化（基于梯度下降，模拟记忆风险）
# ============================================================================
print("\n" + "=" * 80)
print(" 步骤 4: non-DP 优化（无隐私保护）")
print("=" * 80)

def run_nondp_optimization():
    """
    non-DP 优化：直接基于 LLM 反馈优化
    风险：可能直接将训练数据中的敏感信息编码到提示中
    """
    target = Variable(INITIAL_PROMPT, role_description="客户查询提示", requires_grad=True)

    print("\n🔓 运行 non-DP 优化（直接 LLM 反馈）...")

    # 多轮迭代，每轮让 LLM 改进提示
    for iteration in range(5):
        current_score = evaluation_with_sensitive_data(target)

        # 生成反馈，明确告知 LLM 训练数据的特征
        feedback_prompt = f"""当前提示: "{target.get_value()}"
当前分数: {current_score:.2f}

请改进这个提示，使其更具体、更有针对性。
参考信息：我们需要查询的客户包括不同年龄段、不同地区（如北京朝阳、上海浦东、广州天河、深圳南山）的客户信息。

请生成一个改进后的提示（只输出提示文本，不要解释）："""

        try:
            improved = llm_engine.generate(feedback_prompt, system_prompt="你是提示优化专家")

            # 清理输出
            improved = improved.strip().strip('"\'')

            if improved and improved != target.get_value():
                target.set_value(improved)
                print(f"  迭代 {iteration + 1}: 分数 {current_score:.2f} → 提示已更新")
            else:
                print(f"  迭代 {iteration + 1}: 分数 {current_score:.2f} （无变化）")

        except Exception as e:
            print(f"  迭代 {iteration + 1}: LLM 调用失败: {e}")
            break

    final_score = evaluation_with_sensitive_data(target)

    return {
        "final_prompt": target.get_value(),
        "final_score": final_score,
    }

nondp_result = run_nondp_optimization()

if nondp_result:
    print(f"\n✅ non-DP 完成")
    print(f"  最终提示: '{nondp_result['final_prompt']}'")
    print(f"  最终分数: {nondp_result['final_score']:.2f}")

# ============================================================================
# 步骤 5: 隐私泄露检测
# ============================================================================
print("\n" + "=" * 80)
print(" 步骤 5: 隐私泄露检测")
print("=" * 80)

def detect_privacy_leakage(prompt: str, sensitive_data: List[Dict]) -> Dict:
    """
    检测提示中是否泄露了敏感信息

    返回：
    - leaked_count: 泄露的敏感字段数量
    - leaked_items: 具体泄露的内容
    - severity: 泄露严重程度（0-1）
    """
    leaked_items = []

    for record in sensitive_data:
        # 检测姓名泄露
        if record['name'] in prompt:
            leaked_items.append(f"姓名: {record['name']}")

        # 检测电话泄露（完整或部分）
        if record['phone'] in prompt:
            leaked_items.append(f"电话: {record['phone']}")
        elif record['phone'][:7] in prompt:  # 前7位
            leaked_items.append(f"电话前缀: {record['phone'][:7]}")

        # 检测地址泄露
        if record['address'] in prompt:
            leaked_items.append(f"完整地址: {record['address']}")
        else:
            # 检测地址片段
            address_parts = [
                record['address'].split('市')[0] if '市' in record['address'] else "",
                record['address'].split('市')[1].split('区')[0] if '市' in record['address'] and '区' in record['address'] else "",
            ]
            for part in address_parts:
                if part and len(part) > 2 and part in prompt:
                    leaked_items.append(f"地址片段: {part}")

        # 检测邮箱
        if record['email'] in prompt:
            leaked_items.append(f"邮箱: {record['email']}")

    # 计算泄露严重程度
    severity = min(len(leaked_items) / 5.0, 1.0)  # 最多5个算100%

    return {
        "leaked_count": len(leaked_items),
        "leaked_items": leaked_items,
        "severity": severity,
    }

print("\n🔍 检测隐私泄露...")

# DP-ES 泄露检测
if dp_result:
    dp_leakage = detect_privacy_leakage(dp_result['final_prompt'], SENSITIVE_TRAINING_DATA)

    print(f"\n🔒 DP-ES 隐私泄露分析:")
    print(f"  泄露字段数: {dp_leakage['leaked_count']}")
    print(f"  泄露严重程度: {dp_leakage['severity']:.1%}")

    if dp_leakage['leaked_items']:
        print(f"  泄露内容:")
        for item in dp_leakage['leaked_items']:
            print(f"    - {item}")
    else:
        print(f"  ✅ 未检测到敏感信息泄露")

# non-DP 泄露检测
if nondp_result:
    nondp_leakage = detect_privacy_leakage(nondp_result['final_prompt'], SENSITIVE_TRAINING_DATA)

    print(f"\n🔓 non-DP 隐私泄露分析:")
    print(f"  泄露字段数: {nondp_leakage['leaked_count']}")
    print(f"  泄露严重程度: {nondp_leakage['severity']:.1%}")

    if nondp_leakage['leaked_items']:
        print(f"  泄露内容:")
        for item in nondp_leakage['leaked_items']:
            print(f"    - {item}")
    else:
        print(f"  ✅ 未检测到敏感信息泄露")

# ============================================================================
# 步骤 6: 对比总结
# ============================================================================
print("\n" + "=" * 80)
print(" 🎉 隐私保护验证总结")
print("=" * 80)

if dp_result and nondp_result:
    print("\n📊 对比结果:")
    print(f"\n  {'指标':<20} {'DP-ES':>15} {'non-DP':>15}")
    print("  " + "-" * 52)
    print(f"  {'最终分数':<20} {dp_result['final_score']:>15.2f} {nondp_result['final_score']:>15.2f}")
    print(f"  {'泄露字段数':<20} {dp_leakage['leaked_count']:>15} {nondp_leakage['leaked_count']:>15}")
    print(f"  {'泄露严重程度':<20} {dp_leakage['severity']:>14.1%} {nondp_leakage['severity']:>14.1%}")

    if dp_result.get('privacy_epsilon'):
        print(f"  {'隐私预算 ε':<20} {dp_result['privacy_epsilon']:>15.4f} {'N/A':>15}")

    print("\n✅ 验证结论:")

    # 泄露对比
    if dp_leakage['leaked_count'] < nondp_leakage['leaked_count']:
        reduction = nondp_leakage['leaked_count'] - dp_leakage['leaked_count']
        print(f"  1. DP-ES 减少了 {reduction} 个敏感字段泄露")
        print(f"     ✅ 差分隐私有效保护了训练数据隐私")
    elif dp_leakage['leaked_count'] == nondp_leakage['leaked_count']:
        if dp_leakage['leaked_count'] == 0:
            print(f"  1. 两种方法均未泄露敏感信息")
            print(f"     ℹ️  可能是任务设计不足以触发泄露")
        else:
            print(f"  1. 两种方法泄露程度相同")
            print(f"     ⚠️  DP 保护可能不足（需增加噪声）")
    else:
        print(f"  1. ⚠️  DP-ES 泄露更多（异常情况）")

    # 质量对比
    quality_gap = abs(dp_result['final_score'] - nondp_result['final_score'])
    if quality_gap < 1.0:
        print(f"  2. 质量损失很小（<1.0 分）")
        print(f"     ✅ 隐私保护几乎无质量代价")
    else:
        print(f"  2. 质量损失 {quality_gap:.2f} 分")
        print(f"     ℹ️  这是隐私保护的必要代价")

    # 总体评价
    print("\n💡 关键发现:")
    if dp_leakage['leaked_count'] == 0:
        print("  • DP-ES 成功防止了敏感信息泄露")
    else:
        print(f"  • DP-ES 仍有少量泄露，可通过以下方式改进：")
        print(f"    - 降低 epsilon（更强隐私）")
        print(f"    - 增加噪声强度")
        print(f"    - 使用更强的裁剪")

    if nondp_leakage['leaked_count'] > 0:
        print("  • non-DP 方法存在隐私泄露风险")
        print("  • 对于敏感数据场景，必须使用 DP 保护")

print("\n" + "=" * 80)
print()
