"""
DP-TextGrad GSM8K 可扩展测试
支持不同规模的实验配置

用法:
    python test_dp_gsm8k_scalable.py --samples 10 --epsilon 5.0
    python test_dp_gsm8k_scalable.py --samples 100 --epsilon 10.0 --iterations 10
    python test_dp_gsm8k_scalable.py --samples 500 --epsilon 20.0 --save results_500.json
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

from dp_textgrad import Variable, BlackboxLLM
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
)


# ============================================================================
# 配置参数
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="DP-TextGrad GSM8K 可扩展测试")

    # 数据集配置
    parser.add_argument('--samples', type=int, default=10,
                        help='GSM8K 样本数量 (默认: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')

    # 隐私配置
    parser.add_argument('--epsilon', type=float, default=5.0,
                        help='总隐私预算 ε (默认: 5.0)')
    parser.add_argument('--delta', type=float, default=1e-4,
                        help='δ 参数 (默认: 1e-4)')
    parser.add_argument('--composition', choices=['basic', 'advanced'],
                        default='advanced',
                        help='组合定理类型 (默认: advanced)')

    # 优化配置
    parser.add_argument('--iterations', type=int, default=5,
                        help='最大迭代轮数 (默认: 5)')
    parser.add_argument('--population', type=int, default=4,
                        help='种群大小 (默认: 4)')
    parser.add_argument('--parents', type=int, default=2,
                        help='选择的父代数量 (默认: 2)')
    parser.add_argument('--offspring', type=int, default=2,
                        help='每个父代的子代数量 (默认: 2)')

    # 输出配置
    parser.add_argument('--save', type=str, default=None,
                        help='保存结果的文件路径 (默认: 不保存)')
    parser.add_argument('--verbose', action='store_true',
                        help='详细输出模式')

    # 优化选项
    parser.add_argument('--cache', action='store_true',
                        help='启用评分缓存')
    parser.add_argument('--adaptive-clipping', action='store_true',
                        help='启用自适应裁剪')

    return parser.parse_args()


# ============================================================================
# 数据集加载
# ============================================================================

def load_gsm8k_samples(num_samples: int, seed: int = 42) -> List[Dict]:
    """加载 GSM8K 样本 (模拟数据)"""
    random.seed(seed)

    # 真实应用中应从文件加载
    # 这里使用模拟数据作为示例
    samples = [
        {
            "question": f"Math problem {i+1}: Calculate 5 * (10 + {i})",
            "answer": str(5 * (10 + i)),
            "id": f"gsm8k_{i}"
        }
        for i in range(num_samples)
    ]

    return samples


# ============================================================================
# 评估函数
# ============================================================================

def create_evaluation_fn(test_samples: List[Dict], llm):
    """创建评估函数"""

    def evaluate(var: Variable) -> float:
        """评估提示的质量

        在真实场景中，这会:
        1. 使用 var 中的提示
        2. 在测试集上运行
        3. 计算准确率作为分数

        这里简化为模拟评分
        """
        prompt = var.get_value()

        # 模拟评估: 基于提示长度和复杂度
        score = 0.0

        # 基础分: 提示不能太短
        if len(prompt) > 50:
            score += 0.3

        # 包含关键词奖励
        keywords = ["step by step", "calculate", "show work", "explain"]
        for keyword in keywords:
            if keyword.lower() in prompt.lower():
                score += 0.1

        # 添加随机噪声模拟真实评估的不确定性
        score += random.uniform(-0.1, 0.1)

        return max(0.0, min(1.0, score))  # 限制在 [0, 1]

    return evaluate


# ============================================================================
# 主实验流程
# ============================================================================

def run_experiment(args):
    """运行完整实验"""

    print("=" * 80)
    print("DP-TextGrad GSM8K 可扩展测试")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  样本数量: {args.samples}")
    print(f"  隐私预算: ε={args.epsilon}, δ={args.delta}")
    print(f"  组合方式: {args.composition}")
    print(f"  迭代次数: {args.iterations}")
    print(f"  种群大小: {args.population}")
    print(f"  随机种子: {args.seed}")
    print()

    start_time = time.time()

    # ========================================================================
    # 1. 加载数据
    # ========================================================================
    print("[1/6] 加载 GSM8K 数据...")
    samples = load_gsm8k_samples(args.samples, args.seed)
    print(f"✓ 加载 {len(samples)} 个样本")

    # ========================================================================
    # 2. 初始化 LLM (模拟)
    # ========================================================================
    print("\n[2/6] 初始化 LLM...")
    # 在真实场景中使用:
    # llm = BlackboxLLM("gpt-3.5-turbo")
    llm = None  # 这个示例不需要真实 LLM
    print("✓ LLM 初始化完成")

    # ========================================================================
    # 3. 初始化提示
    # ========================================================================
    print("\n[3/6] 初始化优化目标...")
    initial_prompt = """Solve this math problem step by step:
{question}

Show your work and explain each step."""

    target = Variable(
        initial_prompt,
        role_description="Math problem solving prompt",
        requires_grad=True
    )
    print(f"✓ 初始提示长度: {len(initial_prompt)} 字符")

    # ========================================================================
    # 4. 配置 DP 组件
    # ========================================================================
    print("\n[4/6] 配置差分隐私组件...")

    # 计算每个候选的预算分配
    total_evaluations = args.iterations * args.population
    epsilon_per_candidate = args.epsilon / (total_evaluations * 1.2)  # 留 20% 余量

    print(f"  总评估次数: {total_evaluations}")
    print(f"  每候选预算: ε={epsilon_per_candidate:.4f}")

    # DPScorer
    scorer = DPScorer(DPScorerConfig(
        epsilon_per_candidate=epsilon_per_candidate,
        delta_per_candidate=args.delta / total_evaluations,
        composition=args.composition,
        clipping_value=1.0,
        noise_multiplier=1.0,
        enable_score_cache=args.cache,
        adaptive_clipping=args.adaptive_clipping,
    ))

    # DPSelector
    selector = DPSelector(DPSelectorConfig(
        select_k=args.parents,
        epsilon=epsilon_per_candidate * 0.1,  # 选择用 10% 预算
        sensitivity=1.0,
    ))

    # Mutation Engine
    mutation_config = MutationConfig(
        offspring_per_parent=args.offspring,
        allow_identity_offspring=False,
    )

    # 简化的变异函数 (不使用 LLM)
    def simple_mutation_fn(parent, iteration, rng, feedback):
        """简单的提示变异策略"""
        prompt = parent.variable.get_value()

        # 变异策略: 添加或修改指令
        mutations = [
            "Think carefully before answering.\n" + prompt,
            prompt + "\nDouble-check your calculations.",
            prompt.replace("step by step", "step by step, showing all work"),
            prompt + "\nProvide a clear final answer.",
        ]

        children = []
        for _ in range(mutation_config.offspring_per_parent):
            mutated = rng.choice(mutations)
            child = Variable(
                mutated,
                role_description=parent.variable.get_role_description(),
                requires_grad=True
            )
            children.append(child)

        return children

    mutation_engine = MutationEngine(
        mutation_fn=simple_mutation_fn,
        config=mutation_config,
    )

    # Privacy Accountant
    accountant = PrivacyAccountant(
        target_epsilon=args.epsilon,
        target_delta=args.delta
    )

    print("✓ DP 组件配置完成")

    # ========================================================================
    # 5. 运行 DP-ES 优化
    # ========================================================================
    print(f"\n[5/6] 运行 DP-ES 优化 ({args.iterations} 轮)...")

    # 创建评估函数
    evaluation_fn = create_evaluation_fn(samples, llm)

    # 初始评估
    initial_score = evaluation_fn(target)
    print(f"  初始分数: {initial_score:.3f}")

    # 创建策略
    strategy = DPEvolutionStrategy(
        parameter=target,
        evaluation_fn=evaluation_fn,
        scorer=scorer,
        selector=selector,
        mutation_engine=mutation_engine,
        accountant=accountant,
        config=DPEvolutionConfig(
            population_size=args.population,
            parents_to_select=args.parents,
            max_iterations=args.iterations,
            rng_seed=args.seed,
        ),
    )

    # 运行优化
    iteration_scores = [initial_score]

    for i in range(args.iterations):
        print(f"\n  轮次 {i+1}/{args.iterations}:")

        try:
            strategy.step()

            # 评估当前最佳
            current_score = evaluation_fn(target)
            iteration_scores.append(current_score)

            # 显示进度
            epsilon_used = accountant.consumed_epsilon
            epsilon_remaining = args.epsilon - epsilon_used

            print(f"    当前分数: {current_score:.3f}")
            print(f"    隐私预算: 已用 {epsilon_used:.3f}, 剩余 {epsilon_remaining:.3f}")

        except Exception as e:
            print(f"    ⚠️  错误: {e}")
            break

    final_score = iteration_scores[-1]
    improvement = final_score - initial_score

    print(f"\n✓ 优化完成")
    print(f"  最终分数: {final_score:.3f}")
    print(f"  改进幅度: {improvement:+.3f} ({improvement/initial_score*100:+.1f}%)")

    # ========================================================================
    # 6. 生成报告
    # ========================================================================
    print("\n[6/6] 生成结果报告...")

    elapsed_time = time.time() - start_time

    results = {
        "config": {
            "samples": args.samples,
            "epsilon": args.epsilon,
            "delta": args.delta,
            "composition": args.composition,
            "iterations": args.iterations,
            "population": args.population,
            "seed": args.seed,
        },
        "privacy": {
            "epsilon_consumed": accountant.consumed_epsilon,
            "delta_consumed": accountant.consumed_delta,
            "epsilon_remaining": args.epsilon - accountant.consumed_epsilon,
        },
        "performance": {
            "initial_score": initial_score,
            "final_score": final_score,
            "improvement": improvement,
            "improvement_percent": (improvement / initial_score * 100) if initial_score > 0 else 0,
            "iteration_scores": iteration_scores,
        },
        "runtime": {
            "total_seconds": elapsed_time,
            "seconds_per_iteration": elapsed_time / args.iterations if args.iterations > 0 else 0,
        }
    }

    # 保存结果
    if args.save:
        save_path = Path(args.save)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ 结果已保存到: {save_path}")

    # 打印摘要
    print("\n" + "=" * 80)
    print("实验摘要")
    print("=" * 80)
    print(f"样本数量: {args.samples}")
    print(f"初始分数: {initial_score:.3f}")
    print(f"最终分数: {final_score:.3f}")
    print(f"改进幅度: {improvement:+.3f} ({improvement/initial_score*100:+.1f}%)")
    print(f"隐私消耗: ε={accountant.consumed_epsilon:.3f} (目标: {args.epsilon})")
    print(f"运行时间: {elapsed_time:.1f} 秒")
    print("=" * 80)

    return results


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    args = parse_args()

    try:
        results = run_experiment(args)

        # 判断成功/失败
        if results['performance']['improvement'] > 0:
            print("\n✅ 实验成功: 提示有改进!")
            exit(0)
        else:
            print("\n⚠️  实验完成: 提示未改进")
            exit(0)

    except KeyboardInterrupt:
        print("\n\n⚠️  实验被用户中断")
        exit(130)

    except Exception as e:
        print(f"\n\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
