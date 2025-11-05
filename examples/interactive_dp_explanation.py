"""
交互式DP-ES原理演示

这个脚本通过具体数值演示变异算法和差分隐私的工作原理，
帮助你理解整个系统如何运作。

运行：python examples/interactive_dp_explanation.py
"""

import random
import numpy as np
from typing import List, Tuple


# ============================================================================
# 第一部分：变异算法演示
# ============================================================================

def demonstrate_mutation_evolution():
    """演示智能变异如何工作"""

    print("\n" + "="*80)
    print("📚 第一部分：智能变异算法演示")
    print("="*80)

    # 模拟一个简单的prompt优化任务
    print("\n任务：优化一个问答prompt")
    print("目标：提高prompt的清晰度和效果")

    # 父代
    parent_text = "Answer the question."
    print(f"\n🧬 父代文本:")
    print(f"   '{parent_text}'")

    print("\n" + "-"*80)
    print("变异模式对比：")
    print("-"*80)

    # 1. 简单拼接（旧方法）
    print("\n❌ 方法1：简单字符串拼接")
    simple_mutations = [
        parent_text + " Be clear.",
        parent_text + " Provide details.",
    ]
    for i, mut in enumerate(simple_mutations, 1):
        print(f"   后代{i}: '{mut}'")

    print("\n   问题：")
    print("   - 只是机械拼接，没有语义理解")
    print("   - 可能产生不通顺的文本")
    print("   - 搜索空间质量低")

    # 2. LLM引导（新方法）
    print("\n✅ 方法2：LLM引导智能变异")
    print("   LLM理解父代含义，生成有意义的改进：")

    llm_mutations = [
        "Provide a clear, step-by-step answer to the question with examples.",
        "Answer the question thoroughly, explaining your reasoning and supporting evidence.",
    ]
    for i, mut in enumerate(llm_mutations, 1):
        print(f"   后代{i}: '{mut}'")

    print("\n   优势：")
    print("   - 语义连贯、通顺自然")
    print("   - 变异有实际意义")
    print("   - 搜索空间质量高")

    # 3. 自适应变异（新方法增强版）
    print("\n⭐ 方法3：自适应变异（根据阶段调整）")

    phases = [
        ("早期（探索阶段）", "EXPLORE模式 - 大胆创新",
         ["Transform this into a Socratic questioning approach",
          "Reframe as a guided discovery learning prompt"]),
        ("中期（平衡阶段）", "BALANCED模式 - 混合策略",
         ["Answer with clear structure and concrete examples",
          "Provide a comprehensive response with step-by-step reasoning"]),
        ("后期（利用阶段）", "EXPLOIT模式 - 精细调优",
         ["Answer clearly and concisely with examples",
          "Provide a clear, well-structured answer"]),
    ]

    for phase_name, mode, mutations in phases:
        print(f"\n   {phase_name}")
        print(f"   策略：{mode}")
        for i, mut in enumerate(mutations, 1):
            print(f"     后代{i}: '{mut[:60]}...'")

    print("\n" + "-"*80)
    print("💡 关键洞察：")
    print("   智能变异让每一步优化都更有价值，而非盲目搜索！")
    print("="*80)


# ============================================================================
# 第二部分：差分隐私机制演示
# ============================================================================

def demonstrate_dp_mechanism():
    """演示差分隐私如何保护数据"""

    print("\n" + "="*80)
    print("🔒 第二部分：差分隐私保护机制演示")
    print("="*80)

    # 设置
    np.random.seed(42)
    random.seed(42)

    print("\n场景：评估候选prompt在私有数据集上的表现")
    print("私有数据：包含敏感的用户问题和答案")

    # 模拟候选和真实分数
    candidates = [
        "Explain clearly",
        "Provide details",
        "Use examples",
        "Be concise",
    ]

    true_scores = [0.75, 0.82, 0.91, 0.65]

    print(f"\n{'候选':<20} {'真实分数':<15}")
    print("-"*40)
    for cand, score in zip(candidates, true_scores):
        print(f"{cand:<20} {score:<15.3f}")

    # DP参数
    epsilon = 0.5
    delta = 1e-5
    clipping_value = 1.0

    print("\n" + "-"*80)
    print("差分隐私保护三步骤：")
    print("-"*80)

    # 步骤1：裁剪
    print("\n📌 步骤1：裁剪（Clipping）")
    print(f"   目的：限制单条记录的影响")
    print(f"   裁剪范围：[-{clipping_value}, +{clipping_value}]")

    clipped_scores = [min(max(s, -clipping_value), clipping_value) for s in true_scores]

    print(f"\n   {'候选':<20} {'真实分数':<15} {'裁剪后':<15}")
    print("   " + "-"*50)
    for cand, true, clip in zip(candidates, true_scores, clipped_scores):
        changed = "✂️" if true != clip else ""
        print(f"   {cand:<20} {true:<15.3f} {clip:<15.3f} {changed}")

    print(f"\n   敏感度 Δf = 2 × {clipping_value} = {2*clipping_value}")

    # 步骤2：计算噪声尺度
    print("\n📌 步骤2：计算噪声尺度（Noise Scale）")

    sensitivity = 2 * clipping_value
    sigma = (sensitivity / epsilon) * np.sqrt(2 * np.log(1.25 / delta))

    print(f"   公式：σ = (Δf / ε) × √(2 ln(1.25/δ))")
    print(f"   σ = ({sensitivity} / {epsilon}) × √(2 ln(1.25/{delta}))")
    print(f"   σ = {sigma:.4f}")

    # 步骤3：添加噪声
    print("\n📌 步骤3：添加高斯噪声")
    print(f"   从 N(0, σ²) 采样噪声")

    noises = [np.random.normal(0, sigma) for _ in range(len(candidates))]
    dp_scores = [clip + noise for clip, noise in zip(clipped_scores, noises)]

    print(f"\n   {'候选':<20} {'裁剪分数':<15} {'噪声':<15} {'DP分数':<15}")
    print("   " + "-"*65)
    for cand, clip, noise, dp in zip(candidates, clipped_scores, noises, dp_scores):
        print(f"   {cand:<20} {clip:<15.3f} {noise:+15.3f} {dp:<15.3f}")

    print(f"\n   隐私保证：(ε={epsilon}, δ={delta})")

    print("\n" + "-"*80)
    print("🔍 隐私保护效果：")
    print("-"*80)

    # 对比有无隐私保护
    print("\n比较：添加/移除一条敏感记录的影响")
    print("\n不使用DP（无保护）：")
    print("   数据集D（含敏感记录）  → 真实分数 [0.75, 0.82, 0.91, 0.65]")
    print("   数据集D'（移除记录）   → 真实分数 [0.70, 0.85, 0.88, 0.68]")
    print("   差异明显 ❌ → 攻击者可以推断记录是否存在！")

    print("\n使用DP（有保护）：")
    print(f"   数据集D（含敏感记录）  → DP分数 {dp_scores}")

    # 模拟移除记录后的DP分数
    dp_scores_removed = [clip + np.random.normal(0, sigma)
                         for clip in clipped_scores]
    print(f"   数据集D'（移除记录）   → DP分数 {dp_scores_removed}")

    # 计算概率比
    prob_ratio = np.exp(epsilon)
    print(f"\n   概率比界限：Pr[M(D)] / Pr[M(D')] ≤ e^ε = {prob_ratio:.3f}")
    print(f"   差异被噪声掩盖 ✅ → 攻击者难以推断！")

    print("\n" + "="*80)
    print("💡 关键洞察：")
    print("   噪声大小由ε控制：ε越小，噪声越大，隐私越强！")
    print("="*80)


# ============================================================================
# 第三部分：完整DP-ES流程演示
# ============================================================================

def demonstrate_full_dp_es_iteration():
    """演示完整的DP-ES迭代过程"""

    print("\n" + "="*80)
    print("🔄 第三部分：完整DP-ES迭代流程演示")
    print("="*80)

    np.random.seed(123)
    random.seed(123)

    # 初始种群
    population = [
        "Answer the question",
        "Explain clearly",
        "Provide details with examples",
        "Be comprehensive",
    ]

    print("\n初始种群（4个候选）：")
    for i, p in enumerate(population):
        print(f"  {i+1}. '{p}'")

    # 隐私预算
    epsilon_scoring = 0.5
    epsilon_selection = 0.1
    delta = 1e-5

    print(f"\n隐私预算配置：")
    print(f"  - 评分：ε={epsilon_scoring}, δ={delta}")
    print(f"  - 选择：ε={epsilon_selection}, δ=0")

    print("\n" + "─"*80)
    print("第1步：DP评分")
    print("─"*80)

    # 模拟真实评分
    true_scores = [0.45, 0.62, 0.78, 0.53]

    print("\n执行评估函数（使用私有数据）:")
    for i, (cand, score) in enumerate(zip(population, true_scores)):
        print(f"  候选{i+1}: {score:.3f}")

    # DP评分
    clipping = 1.0
    clipped = [min(s, clipping) for s in true_scores]
    sigma = (2*clipping / epsilon_scoring) * np.sqrt(2*np.log(1.25/delta))
    noises = [np.random.normal(0, sigma) for _ in range(len(population))]
    dp_scores = [c + n for c, n in zip(clipped, noises)]

    print(f"\nDP处理（σ={sigma:.2f}）:")
    print(f"  {'候选':<5} {'真实':<8} {'裁剪':<8} {'噪声':<10} {'DP分数':<10}")
    print("  " + "-"*50)
    for i, (s, c, n, dp) in enumerate(zip(true_scores, clipped, noises, dp_scores), 1):
        print(f"  {i:<5} {s:<8.3f} {c:<8.3f} {n:+10.3f} {dp:<10.3f}")

    print(f"\n✅ 隐私消耗：ε={epsilon_scoring}, δ={delta}")

    print("\n" + "─"*80)
    print("第2步：DP选择")
    print("─"*80)

    # 选择top-2
    select_k = 2
    gumbel_scale = 1.0 / epsilon_selection

    print(f"\n目标：选择top-{select_k}个候选")
    print(f"方法：添加Gumbel噪声（scale={gumbel_scale:.1f}）")

    gumbel_noises = [np.random.gumbel(0, gumbel_scale) for _ in range(len(population))]
    noisy_scores = [dp + g for dp, g in zip(dp_scores, gumbel_noises)]

    print(f"\n  {'候选':<5} {'DP分数':<10} {'Gumbel噪声':<15} {'加噪分数':<12} {'选中':<5}")
    print("  " + "-"*60)

    sorted_indices = np.argsort(noisy_scores)[::-1]
    selected_indices = sorted_indices[:select_k]

    for i, (dp, g, ns) in enumerate(zip(dp_scores, gumbel_noises, noisy_scores)):
        selected = "✓" if i in selected_indices else ""
        print(f"  {i+1:<5} {dp:<10.3f} {g:+15.3f} {ns:<12.3f} {selected:<5}")

    selected_candidates = [population[i] for i in selected_indices]
    print(f"\n选中的父代：")
    for i, cand in enumerate(selected_candidates, 1):
        print(f"  父代{i}: '{cand}'")

    print(f"\n✅ 隐私消耗：ε={epsilon_selection}, δ=0")
    print(f"✅ 累计隐私：ε={epsilon_scoring + epsilon_selection}, δ={delta}")

    print("\n" + "─"*80)
    print("第3步：智能变异")
    print("─"*80)

    print("\n策略：自适应变异（当前阶段：EXPLORE）")
    print("每个父代生成2个后代\n")

    offspring = []
    for i, parent in enumerate(selected_candidates, 1):
        print(f"父代{i}: '{parent}'")
        # 模拟LLM生成的变异
        children = [
            f"{parent} with clear examples and reasoning",
            f"{parent} step by step with detailed explanation",
        ]
        offspring.extend(children)
        for j, child in enumerate(children, 1):
            print(f"  ├─ 后代{i}.{j}: '{child}'")

    print(f"\n✅ 隐私消耗：0（变异不使用私有数据）")
    print(f"✅ 累计隐私：ε={epsilon_scoring + epsilon_selection}, δ={delta} (不变)")

    print("\n" + "─"*80)
    print("第4步：构建新种群")
    print("─"*80)

    new_population = selected_candidates + offspring
    print(f"\n合并：{len(selected_candidates)}个父代 + {len(offspring)}个后代 = {len(new_population)}个")

    # 选择top-4进入下一代
    print(f"保留top-4进入下一代")

    print("\n" + "="*80)
    print("✅ 迭代1完成！")
    print(f"   - 种群更新：{len(population)} → {len(new_population[:4])}个候选")
    print(f"   - 隐私消耗：ε={epsilon_scoring + epsilon_selection}, δ={delta}")
    print(f"   - 预计需要{3.0/(epsilon_scoring + epsilon_selection):.1f}次迭代耗尽ε=3.0预算")
    print("="*80)


# ============================================================================
# 第四部分：隐私泄露风险对比
# ============================================================================

def demonstrate_privacy_leakage_risk():
    """演示有无DP保护时的隐私泄露风险"""

    print("\n" + "="*80)
    print("⚠️  第四部分：隐私泄露风险对比")
    print("="*80)

    print("\n场景设定：")
    print("  训练数据包含敏感记录：'Patient X has rare disease Y'")
    print("  任务：优化医疗问答prompt")

    print("\n" + "─"*80)
    print("❌ 情况1：不使用差分隐私")
    print("─"*80)

    print("\n数据集D（包含敏感记录）：")
    print("  优化结果：'When diagnosing disease Y, check for symptoms X showed'")
    print("  ↑ 明确泄露了患者X的信息！")

    print("\n数据集D'（移除敏感记录）：")
    print("  优化结果：'Follow standard diagnostic protocols'")
    print("  ↑ 完全不同的结果")

    print("\n攻击者推断：")
    print("  结果差异巨大 → 敏感记录必定在训练集中！")
    print("  隐私泄露风险：⚠️⚠️⚠️ 高")

    print("\n" + "─"*80)
    print("✅ 情况2：使用差分隐私（ε=1.0）")
    print("─"*80)

    print("\n数据集D（包含敏感记录）：")
    print("  DP-ES优化结果：'Provide comprehensive diagnosis with evidence'")

    print("\n数据集D'（移除敏感记录）：")
    print("  DP-ES优化结果：'Provide detailed diagnosis based on symptoms'")

    print("\n结果相似度分析：")
    similarity = 0.85
    print(f"  词汇重叠率：{similarity:.1%}")
    print(f"  语义相似度：高")

    epsilon = 1.0
    max_prob_ratio = np.exp(epsilon)
    print(f"\n差分隐私保证：")
    print(f"  Pr[结果1] / Pr[结果2] ≤ e^{epsilon} = {max_prob_ratio:.2f}")
    print(f"  即使在最坏情况下，概率比也被限制在{max_prob_ratio:.2f}倍")

    print(f"\n攻击者推断成功率：")
    baseline = 0.5  # 随机猜测
    advantage = (max_prob_ratio - 1) / (max_prob_ratio + 1)
    attack_success = baseline + advantage/2
    print(f"  理论上界：≤ {attack_success:.1%}")
    print(f"  vs 随机猜测：50%")
    print(f"  隐私泄露风险：✅ 低（受控）")

    print("\n" + "="*80)
    print("💡 关键洞察：")
    print("   ε参数权衡：")
    print("   - ε小 → 强隐私，但性能可能下降")
    print("   - ε大 → 弱隐私，但性能接近非DP")
    print("   - 推荐：ε ∈ [0.5, 3.0] 平衡隐私与效用")
    print("="*80)


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行所有演示"""

    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*20 + "DP-ES 技术原理交互式演示" + " "*20 + "║")
    print("╚" + "═"*78 + "╝")

    print("\n本演示分为4个部分：")
    print("  1️⃣  智能变异算法演示")
    print("  2️⃣  差分隐私保护机制演示")
    print("  3️⃣  完整DP-ES迭代流程演示")
    print("  4️⃣  隐私泄露风险对比")

    input("\n按Enter键开始演示...")

    # 第1部分
    demonstrate_mutation_evolution()
    input("\n按Enter键继续下一部分...")

    # 第2部分
    demonstrate_dp_mechanism()
    input("\n按Enter键继续下一部分...")

    # 第3部分
    demonstrate_full_dp_es_iteration()
    input("\n按Enter键继续下一部分...")

    # 第4部分
    demonstrate_privacy_leakage_risk()

    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*25 + "演示结束，谢谢观看！" + " "*25 + "║")
    print("╚" + "═"*78 + "╝")

    print("\n📚 进一步学习：")
    print("  - 详细文档：docs/DP_ES_TECHNICAL_EXPLANATION.md")
    print("  - 变异策略：dp-textgrad/dp_es/MUTATION_STRATEGIES.md")
    print("  - 隐私验证：evaluation/README_PRIVACY_VERIFICATION.md")
    print()


if __name__ == "__main__":
    main()
