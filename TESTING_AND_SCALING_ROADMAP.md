# DP-TextGrad 测试流程与实验扩展路线图

**日期**: 2025-12-16
**状态**: Phase 1-4 修复完成，准备系统测试和扩大规模

---

## 📋 总体目标

1. ✅ **验证所有修复** - 确保 Phase 1-4 的隐私修复都正确工作
2. 🎯 **系统测试** - 完整的端到端测试
3. 📈 **扩大实验规模** - 从玩具实验到真实规模
4. 📊 **性能评估** - 隐私-效用权衡分析

---

## 第一阶段: 基础验证测试 (预计 30 分钟)

### Step 1.1: 核心单元测试
**目的**: 验证 DP 机制的正确性

```bash
# 1. DP-ES 核心组件测试 (已修复)
python -m pytest tests/test_dp_es.py -v

# 2. 基础 API 测试
python -m pytest tests/test_api.py -v
python -m pytest tests/test_basics.py -v
python -m pytest tests/test_engines.py -v
```

**预期结果**: 所有测试通过 ✅

**验证点**:
- ✅ DPScorer 正确裁剪和加噪
- ✅ 隐私预算使用组合定理正确计算
- ✅ DPSelector 正确使用 Gumbel 机制
- ✅ PrivacyAccountant 正确跟踪预算

---

### Step 1.2: Phase 2 修复验证
**目的**: 确认隐私预算和反馈修复

```bash
# 验证组合定理实现
python test_phase2_simple.py

# 验证反馈默认禁用
python test_phase2_privacy_fixes.py
```

**预期结果**:
- ✅ 基础组合: ε_total = k × ε₀
- ✅ 高级组合: ε_total ≈ √(2k ln(1/δ)) × ε₀
- ✅ 反馈默认禁用，启用时显示警告

---

### Step 1.3: Phase 4 DPScoreRecord 泄露修复验证
**目的**: 确认生产模式下无数据泄露

```bash
# 验证 records 在生产模式下为空
python test_debug_mode_fix.py
```

**预期结果**:
- ✅ 生产模式: `scores.records == []` (无泄露)
- ✅ 调试模式: `scores.records` 有数据但显示警告
- ✅ 正常操作 (dp_score, noise_magnitude) 仍然可用

---

### Step 1.4: Phase 3 严格审计测试
**目的**: 验证 DP 实现的统计正确性

```bash
# 运行严格审计框架
python test_strict_audit.py
```

**预期结果**:
- ✅ 噪声分布审计: K-S 检验通过
- ✅ 隐私损失分布: Pr[loss > ε] ≤ δ (95% 置信区间)
- 🟡 似然比测试: 需要进一步优化

**如果失败**: 这表明 DP 实现可能有问题，需要修复

---

## 第二阶段: 集成测试 (预计 1-2 小时)

### Step 2.1: 最小工作示例
**目的**: 验证 DP-ES 端到端工作流程

```bash
# 最小测试 (快速验证)
python test_dp_minimal_working.py
```

**预期结果**:
- ✅ 能够初始化 DPEvolutionStrategy
- ✅ 能够运行多轮优化
- ✅ 隐私预算正确消耗
- ✅ 参数有所改进

---

### Step 2.2: DP vs Non-DP 对比
**目的**: 验证 DP 确实在保护隐私（有效用损失）

```bash
# 对比测试
python test_dp_vs_nondp_comparison.py
```

**预期结果**:
- ✅ Non-DP 版本: 更高准确率（无隐私保护）
- ✅ DP 版本: 稍低准确率（有隐私保护）
- ✅ ε 越小，效用损失越大（符合理论）

**关键指标**:
```
ε=∞ (Non-DP): 准确率 ~95%
ε=5.0:        准确率 ~85%
ε=2.0:        准确率 ~75%
ε=1.0:        准确率 ~60%
```

---

### Step 2.3: GSM8K 数学题测试 (小规模)
**目的**: 在真实任务上验证功能

```bash
# GSM8K 测试 (10 题，快速验证)
python test_dp_gsm8k_v3.py
```

**配置**:
- 数据集: GSM8K (小学数学题)
- 样本数: 10 题
- 迭代轮数: 3-5 轮
- 预算: ε=5.0, δ=1e-4

**预期结果**:
- ✅ 能够完成优化流程
- ✅ 隐私预算正确消耗
- ✅ 至少 2-3 题答案有改进
- ⚠️ 准确率可能较低（样本太少）

---

### Step 2.4: 隐私-效用曲线
**目的**: 系统性评估隐私代价

```bash
# 隐私效用权衡分析
python test_privacy_utility_curve.py
# 或鲁棒版本
python test_privacy_utility_curve_robust.py
```

**测试点**: ε = [0.5, 1.0, 2.0, 5.0, 10.0, ∞]

**预期结果**:
- ✅ 单调性: ε 越大，准确率越高
- ✅ 渐近性: ε → ∞ 时接近 Non-DP 性能
- ✅ 可视化: 生成隐私-效用曲线图

---

## 第三阶段: 扩大实验规模 (预计 4-8 小时)

### Step 3.1: 增加 GSM8K 样本量

**当前状态**: 10 题（玩具实验）
**目标规模**: 100 → 500 → 1000 题

#### Step 3.1.1: 中等规模 (100 题)

**修改 test_dp_gsm8k_v3.py**:
```python
# 修改配置
NUM_SAMPLES = 100  # 从 10 增加到 100
MAX_ITERATIONS = 5  # 保持不变
EPSILON_BUDGET = 10.0  # 适当增加预算
```

**运行**:
```bash
python test_dp_gsm8k_v3.py 2>&1 | tee results_100_samples.log
```

**预期耗时**: ~1-2 小时（取决于 LLM API 速度）

**关键指标**:
- 初始准确率: ~30-40%
- 最终准确率: ~50-60%
- 改进幅度: +20%
- 隐私预算消耗: ≤ 10.0

---

#### Step 3.1.2: 大规模 (500 题)

**配置**:
```python
NUM_SAMPLES = 500
MAX_ITERATIONS = 10
EPSILON_BUDGET = 20.0
```

**优化**:
1. **并行化**: 使用多进程评估候选
2. **缓存**: 启用 `enable_score_cache=True`
3. **批处理**: 批量调用 LLM API

**运行**:
```bash
# 使用后台运行 + 日志
nohup python test_dp_gsm8k_large_scale.py > results_500.log 2>&1 &
```

**预期耗时**: ~4-6 小时

---

#### Step 3.1.3: 完整规模 (1000+ 题)

**配置**:
```python
NUM_SAMPLES = 1000
MAX_ITERATIONS = 15
EPSILON_BUDGET = 50.0
```

**额外优化**:
1. **分布式执行**: 多机并行
2. **检查点保存**: 定期保存中间结果
3. **成本控制**: 使用更便宜的 LLM (如 gpt-3.5-turbo)

---

### Step 3.2: 多种隐私预算对比

**目标**: 在相同数据集上测试不同 ε 值

**实验设计**:
```python
EPSILON_VALUES = [1.0, 2.0, 5.0, 10.0, 20.0]
NUM_SAMPLES = 200  # 固定样本数
MAX_ITERATIONS = 10  # 固定轮数

for eps in EPSILON_VALUES:
    run_experiment(epsilon=eps, samples=NUM_SAMPLES)
    save_results(f"results_eps_{eps}.json")
```

**分析**:
- 绘制 ε-准确率曲线
- 计算隐私代价 (准确率损失 vs ε)
- 推荐最优 ε 值

---

### Step 3.3: 不同组合定理对比

**目标**: 比较 basic vs advanced composition

**实验**:
```python
COMPOSITIONS = ["basic", "advanced"]
NUM_QUERIES = [10, 50, 100, 500]

for comp in COMPOSITIONS:
    for k in NUM_QUERIES:
        epsilon_consumed = compute_composition(
            epsilon_per_query=0.5,
            num_queries=k,
            composition=comp
        )
        print(f"{comp}, k={k}: ε_total = {epsilon_consumed}")
```

**预期结果**:
- Basic: ε_total = k × 0.5 (线性增长)
- Advanced: ε_total ≈ √(2k ln(1/δ)) × 0.5 (次线性)
- 推荐: k 较大时使用 advanced

---

### Step 3.4: 不同任务类型

**扩展到其他任务**:

1. **代码生成**: HumanEval 数据集
2. **摘要生成**: CNN/DailyMail
3. **问答**: SQuAD
4. **翻译**: WMT

**每个任务**:
- 样本数: 100-500
- 评估指标: 任务特定 (BLEU, ROUGE, Exact Match)
- 隐私预算: ε=5.0-10.0

---

## 第四阶段: 高级优化 (可选，预计 2-4 小时)

### Step 4.1: 自适应裁剪优化

**当前**: 固定 `clipping_value`
**改进**: 启用 `adaptive_clipping=True`

```python
config = DPScorerConfig(
    adaptive_clipping=True,  # 启用
    clipping_value=1.0,      # 初始值
    ...
)
```

**效果**: 减少信息损失，提高效用

---

### Step 4.2: Moments Accountant

**当前**: Basic/Advanced composition
**改进**: 实现 Moments Accountant (更紧的界)

**预期收益**:
- ε 节省 20-30%
- 允许更多迭代或更大种群

---

### Step 4.3: 分层采样

**问题**: 当前均匀采样，可能浪费预算在低质量候选上

**改进**: 根据 dp_score 分层分配预算
- 高分候选: 更多评估次数
- 低分候选: 早期淘汰

---

## 第五阶段: 论文级实验 (如果需要发表)

### Step 5.1: 完整基准测试

**数据集**:
- GSM8K: 1319 题 (全部)
- HumanEval: 164 题
- MMLU: 多选题 (抽样 1000 题)

**对比方法**:
- Baseline: Non-DP TextGrad
- DP-TextGrad (本项目)
- DP-SGD (如果适用)
- 其他 DP 优化方法

---

### Step 5.2: 消融实验

**研究问题**:
1. 各个 DP 组件的影响？
   - 仅 DPScorer
   - 仅 DPSelector
   - 完整 DP-ES
2. 不同变异策略的影响？
3. 种群大小的影响？

---

### Step 5.3: 真实数据隐私审计

**使用真实敏感数据** (如医疗、金融数据):
1. 运行完整 DP-ES 优化
2. 使用 `StrictDPAuditor` 进行统计审计
3. 生成审计报告证明隐私保证

---

## 📊 测试清单总结

### 立即执行 (今天)
- [ ] **Step 1.1**: 核心单元测试
- [ ] **Step 1.2**: Phase 2 验证
- [ ] **Step 1.3**: Phase 4 验证
- [ ] **Step 1.4**: Phase 3 审计
- [ ] **Step 2.1**: 最小工作示例
- [ ] **Step 2.2**: DP vs Non-DP 对比

### 短期 (本周)
- [ ] **Step 2.3**: GSM8K 小规模 (10 题)
- [ ] **Step 2.4**: 隐私-效用曲线
- [ ] **Step 3.1.1**: GSM8K 中等规模 (100 题)
- [ ] **Step 3.2**: 多种 ε 值对比

### 中期 (下周)
- [ ] **Step 3.1.2**: GSM8K 大规模 (500 题)
- [ ] **Step 3.3**: 组合定理对比
- [ ] **Step 3.4**: 其他任务类型
- [ ] **Step 4.1**: 自适应裁剪优化

### 长期 (如果需要)
- [ ] **Step 3.1.3**: 完整 GSM8K (1000+ 题)
- [ ] **Step 4.2**: Moments Accountant
- [ ] **Step 5**: 论文级实验

---

## 🚀 快速开始脚本

创建自动化测试脚本:

```bash
#!/bin/bash
# run_all_tests.sh

echo "================================"
echo "DP-TextGrad 完整测试流程"
echo "================================"

# Phase 1: 单元测试
echo -e "\n[1/6] 单元测试..."
python -m pytest tests/test_dp_es.py -v
python -m pytest tests/test_api.py -v

# Phase 2: 修复验证
echo -e "\n[2/6] Phase 2-4 修复验证..."
python test_phase2_simple.py
python test_debug_mode_fix.py

# Phase 3: 审计
echo -e "\n[3/6] 严格审计..."
python test_strict_audit.py

# Phase 4: 集成测试
echo -e "\n[4/6] 最小工作示例..."
python test_dp_minimal_working.py

# Phase 5: 对比测试
echo -e "\n[5/6] DP vs Non-DP..."
python test_dp_vs_nondp_comparison.py

# Phase 6: GSM8K
echo -e "\n[6/6] GSM8K 测试..."
python test_dp_gsm8k_v3.py

echo -e "\n================================"
echo "所有测试完成!"
echo "================================"
```

**运行**:
```bash
chmod +x run_all_tests.sh
./run_all_tests.sh 2>&1 | tee test_results.log
```

---

## 📈 预期结果与成功标准

### 基础验证 (必须通过)
- ✅ 所有单元测试通过
- ✅ 隐私预算计算正确
- ✅ 无数据泄露 (records 为空)
- ✅ 统计审计通过 (噪声分布、隐私损失)

### 功能验证 (应该通过)
- ✅ DP vs Non-DP 有明显效用差距
- ✅ ε 越小，隐私保护越强（效用越低）
- ✅ GSM8K 至少 20% 题目有改进

### 规模化验证 (目标)
- 🎯 100 题: 50-60% 准确率
- 🎯 500 题: 60-70% 准确率
- 🎯 隐私-效用曲线单调递增

### 论文级指标 (如果发表)
- 📊 完整 GSM8K (1319 题): >70% 准确率
- 📊 隐私审计通过 (95% 置信区间)
- 📊 与 baseline 对比有统计显著性

---

**创建日期**: 2025-12-16
**更新者**: Claude Code AI
**状态**: 待执行
