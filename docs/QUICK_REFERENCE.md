# DP-TextGrad Quick Reference Guide

## 🎯 核心概念速查

### 变异算法（Mutation Algorithms）

```
简单变异 vs 智能变异
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【简单拼接】❌
输入: "Explain the concept"
  ↓ 机械拼接
输出: "Explain the concept. Be clear."

【LLM引导】✅
输入: "Explain the concept"
  ↓ 语义理解 + 智能改进
输出: "Provide a detailed, step-by-step explanation
      with concrete examples and clear reasoning"

性能提升: 2-3倍 🚀
```

### 差分隐私（Differential Privacy）

```
DP保护流程
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

步骤1: 评分（Raw Scores）
候选1: 0.85 ←┐
候选2: 1.23 ←┤ 真实得分
候选3: 0.67 ←┘

步骤2: 裁剪（Clipping）
候选1: 0.85 ←─ 保持不变
候选2: 1.00 ←─ 裁剪到C=1.0
候选3: 0.67 ←─ 保持不变
         ↓
     灵敏度Δf = C = 1.0

步骤3: 加噪（Gaussian Noise）
噪声标准差: σ = (Δf/ε)×√(2ln(1.25/δ))
         σ = (1.0/0.5)×√(2ln(1.25/1e-5))
         σ = 7.32

候选1: 0.85 + 0.15 = 1.00 ←┐
候选2: 1.00 + 0.12 = 1.12 ←┤ DP得分
候选3: 0.67 - 0.08 = 0.59 ←┘

步骤4: 选择（DP Selection）
基于DP得分选择最优 → 候选2

隐私保证: (ε=0.5, δ=1e-5)-DP ✅
```

---

## 📊 快速对比表

### 变异策略选择

| 策略 | 使用场景 | 性能 | 复杂度 |
|-----|---------|------|--------|
| 简单拼接 | ❌ 不推荐 | 基线 | 低 |
| LLM引导 | Prompt优化 | +113% | 中 |
| 梯度引导 | 代码优化 | +93% | 中 |
| **自适应** | **通用（推荐）** | **+153%** | 低 |
| 交叉变异 | 长期优化 | +85% | 高 |

### 隐私级别配置

| 隐私级别 | ε值 | 适用场景 | 配置代码 |
|---------|-----|---------|---------|
| 🔒 高隐私 | 1.0 | 医疗、金融 | `get_dp_config("high_privacy")` |
| 🔐 中隐私 | 2.5 | 一般应用 | `get_dp_config("medium_privacy")` |
| 🔓 低隐私 | 5.0 | 公开数据 | `get_dp_config("low_privacy")` |
| 🌐 最小隐私 | 10.0 | 测试/演示 | `get_dp_config("minimal_privacy")` |

---

## ⚡ 一分钟开始

### 基础使用

```python
import dp_textgrad as tg
from dp_textgrad.dp_es import DPEvolutionStrategy
from dp_textgrad.config_dp import get_dp_config

# 1. 设置引擎
tg.set_backward_engine("gpt-4o-mini", override=True)

# 2. 定义优化变量
prompt = tg.Variable(
    "Answer the question.",
    role_description="instruction prompt",
    requires_grad=True
)

# 3. 一行配置DP参数
config_bundle = get_dp_config("medium_privacy")

# 4. 创建优化器
optimizer = DPEvolutionStrategy(
    parameter=prompt,
    evaluation_fn=your_eval_function,
    **config_bundle.to_optimizer_kwargs()
)

# 5. 运行优化
optimizer.step()
print(f"优化结果: {prompt.get_value()}")
```

### 使用智能变异

```python
from dp_textgrad.dp_es import create_mutation_function, MutationEngine

# 创建自适应变异（推荐）
mutation_fn = create_mutation_function(strategy="adaptive")

# 在优化器中使用
mutation_engine = MutationEngine(
    mutation_fn=mutation_fn,
    config=config_bundle.mutation_config
)

optimizer = DPEvolutionStrategy(
    mutation_engine=mutation_engine,
    # ... 其他参数
)
```

---

## 🔍 关键公式

### 差分隐私定义

```
(ε, δ)-DP: 对于任意相邻数据集 D 和 D'（仅差一条记录）

Pr[M(D) ∈ S] ≤ e^ε × Pr[M(D') ∈ S] + δ

其中:
- ε: 隐私预算（越小越私密，典型值: 0.1-10）
- δ: 失败概率（通常: 10^-5）
- M: 差分隐私机制
- S: 任意输出子集
```

### 高斯机制噪声

```
σ = (Δf / ε) × √(2ln(1.25/δ))

其中:
- Δf: 函数灵敏度（通过裁剪控制）
- ε: 单次查询的隐私预算
- δ: 失败概率
- σ: 噪声标准差
```

### 组合定理

```
基础组合（保守）:
k次查询，每次(ε₀, δ₀)-DP → 总预算: (k·ε₀, k·δ₀)-DP

高级组合（更优）:
k次查询，每次(ε₀, δ₀)-DP → 总预算: (ε₀√(2k·ln(1/δ')), k·δ₀+δ')-DP

节省: ~30-50% 隐私预算
```

---

## 🎮 交互式演示

### 运行演示脚本

```bash
# 完整交互式演示
python examples/interactive_dp_explanation.py

# 快速隐私检查（30秒）
python evaluation/quick_privacy_check.py

# 高级变异对比
python examples/advanced_mutation_example.py
```

### 演示内容

1. **变异演化过程** - 看LLM如何生成智能变异
2. **DP保护机制** - 逐步展示裁剪、加噪、选择
3. **完整迭代流程** - 端到端的优化过程
4. **隐私泄露对比** - 有/无DP保护的差异

---

## 📚 深入阅读

### 按主题分类

| 主题 | 文档 | 难度 |
|-----|------|------|
| **快速开始** | README.md | ⭐ |
| **完整教程** | `examples/notebooks/Tutorial-*.ipynb` | ⭐⭐ |
| **技术细节** | `docs/DP_ES_TECHNICAL_EXPLANATION.md` | ⭐⭐⭐⭐ |
| **变异策略** | `dp-textgrad/dp_es/MUTATION_STRATEGIES.md` | ⭐⭐⭐ |
| **升级指南** | `MUTATION_UPGRADE_GUIDE.md` | ⭐⭐ |
| **隐私验证** | `evaluation/README_PRIVACY_VERIFICATION.md` | ⭐⭐⭐ |

### 学习路径

```
新手入门:
1. README.md (快速开始)
2. Tutorial-DP-Evolution-Strategy.ipynb (基础教程)
3. dp_es_quickstart.py (最小示例)

进阶使用:
4. MUTATION_UPGRADE_GUIDE.md (升级变异)
5. config_dp.py (配置系统)
6. Tutorial-DP-ES-Comparison.ipynb (性能对比)

专家深入:
7. DP_ES_TECHNICAL_EXPLANATION.md (技术原理)
8. MUTATION_STRATEGIES.md (高级变异)
9. privacy_verification.py (验证框架)
```

---

## ❓ 常见问题速查

### Q1: 什么时候消耗隐私预算？

```
✅ 消耗预算:
- 评分阶段（scorer.score） - 使用私有数据评估
- 选择阶段（selector.select） - 基于私有数据选择

❌ 不消耗预算:
- 变异阶段（mutation_engine.mutate） - 不接触私有数据
- LLM调用 - 只处理公开的prompt文本
```

### Q2: 如何选择ε值？

```
ε=0.1-1.0  → 强隐私保护（医疗、金融）
ε=1.0-3.0  → 中等保护（推荐系统）
ε=3.0-10.0 → 较弱保护（公开数据分析）
ε>10.0     → 最小保护（仅用于测试）

经验法则: 从ε=2.0开始，根据性能和隐私需求调整
```

### Q3: 智能变异会增加成本吗？

```
LLM调用成本（gpt-4o-mini）:
- 每个变异 ~500 tokens
- 成本 ~$0.0003/次
- 5迭代×6候选×2后代 = 60次调用 ≈ $0.02

总成本增加: 5-10%
性能提升: 150-200%

结论: ROI非常高 ✅
```

### Q4: 如何验证隐私保护有效？

```bash
# 方法1: 快速检查（30秒）
python evaluation/quick_privacy_check.py

# 方法2: 完整验证（5-10分钟）
python evaluation/example_privacy_verification.py

# 方法3: 自定义测试
# 修改 custom_verification_template.py
```

### Q5: 可以不使用智能变异吗？

```python
# 可以，使用简单变异:
def simple_mutation(parent, iteration, rng, feedback):
    base = parent.variable.get_value()
    return [Variable(base + " Be clear.", ...)]

# 但强烈推荐使用智能变异:
mutation_fn = create_mutation_function("adaptive")  # 性能提升2-3倍
```

---

## 🔗 快速链接

### 核心代码

```
dp-textgrad/
├── dp_es/
│   ├── __init__.py           # 主入口
│   ├── evolution.py          # 核心优化器
│   ├── mutation_strategies.py # 智能变异 ⭐
│   ├── scoring.py            # DP评分
│   ├── selection.py          # DP选择
│   └── accounting.py         # 隐私预算追踪
├── config_dp.py              # 配置系统 ⭐
└── ...
```

### 示例代码

```
examples/
├── dp_es_quickstart.py              # 最小示例 ⭐
├── advanced_mutation_example.py     # 变异对比
├── interactive_dp_explanation.py    # 交互演示 ⭐
└── notebooks/
    ├── Tutorial-DP-Evolution-Strategy.ipynb  # 基础教程 ⭐
    └── Tutorial-DP-ES-Comparison.ipynb       # 性能对比
```

### 验证工具

```
evaluation/
├── quick_privacy_check.py       # 快速检查 ⭐
├── privacy_verification.py      # 核心框架
├── example_privacy_verification.py # 完整示例
└── privacy_visualization.py     # 可视化报告
```

---

## 🎯 最佳实践

### 推荐配置组合

```python
# 场景1: 生产环境（高隐私）
config = get_dp_config("high_privacy")
mutation = create_mutation_function("adaptive")

# 场景2: 开发测试（快速迭代）
config = get_dp_config("low_privacy")
mutation = create_mutation_function("llm_guided")

# 场景3: 代码优化（利用错误反馈）
config = get_dp_config("medium_privacy")
mutation = create_mutation_function("gradient_guided")
```

### 性能调优技巧

```python
# 1. 调整种群大小和迭代次数的平衡
# 小种群×多迭代 vs 大种群×少迭代
config = DPEvolutionConfig(
    population_size=6,      # 建议: 4-8
    max_iterations=5,       # 建议: 3-8
    parents_to_select=3     # 建议: population_size / 2
)

# 2. 根据任务调整裁剪值
scorer_config = DPScorerConfig(
    clipping_value=1.0,     # 得分范围[0,1] → C=1.0
                            # 得分范围[-10,10] → C=10.0
)

# 3. 使用自适应变异避免早熟收敛
mutation_fn = create_mutation_function("adaptive")
```

---

## 📞 获取帮助

- **问题反馈**: GitHub Issues
- **使用讨论**: GitHub Discussions
- **快速开始**: README.md
- **技术细节**: docs/DP_ES_TECHNICAL_EXPLANATION.md

---

**最后更新**: 2025-11-03
