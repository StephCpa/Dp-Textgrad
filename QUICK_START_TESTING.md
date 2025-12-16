# DP-TextGrad 测试快速开始指南

**更新日期**: 2025-12-16
**状态**: Phase 1-4 修复完成 ✅

---

## 🚀 立即开始

### 方式一: 一键运行所有测试

```bash
# 运行完整测试套件
./run_all_tests.sh
```

这会自动运行所有测试并生成报告。

---

### 方式二: 按阶段运行

#### **阶段 1: 验证修复 (5 分钟)**

```bash
# 验证所有 Phase 1-4 的修复
python -m pytest tests/test_dp_es.py -v        # 核心 DP 组件
python test_phase2_simple.py                   # 隐私预算修复
python test_debug_mode_fix.py                  # DPScoreRecord 修复
```

**预期**: 所有测试通过 ✅

---

#### **阶段 2: 基础功能测试 (10 分钟)**

```bash
# 最小工作示例
python test_dp_minimal_working.py

# DP vs Non-DP 对比
python test_dp_vs_nondp_comparison.py
```

**预期**: DP 版本准确率略低于 Non-DP（证明隐私保护有效）

---

#### **阶段 3: 小规模实验 (30 分钟)**

```bash
# GSM8K 10 题快速测试
python test_dp_gsm8k_scalable.py --samples 10 --epsilon 5.0
```

**预期**: 至少 2-3 题有改进

---

#### **阶段 4: 扩大规模 (1-4 小时)**

```bash
# 100 题中等规模
python test_dp_gsm8k_scalable.py \
    --samples 100 \
    --epsilon 10.0 \
    --iterations 10 \
    --save results_100.json

# 500 题大规模
python test_dp_gsm8k_scalable.py \
    --samples 500 \
    --epsilon 20.0 \
    --iterations 15 \
    --cache \
    --adaptive-clipping \
    --save results_500.json
```

---

## 📊 详细测试选项

### test_dp_gsm8k_scalable.py 参数

```bash
# 基础参数
--samples N          # 样本数量 (默认: 10)
--epsilon E          # 隐私预算 (默认: 5.0)
--delta D            # δ 参数 (默认: 1e-4)
--iterations I       # 迭代轮数 (默认: 5)

# 优化选项
--composition TYPE   # basic 或 advanced (默认: advanced)
--cache              # 启用评分缓存
--adaptive-clipping  # 启用自适应裁剪

# 种群配置
--population P       # 种群大小 (默认: 4)
--parents N          # 父代数量 (默认: 2)
--offspring N        # 子代数量 (默认: 2)

# 其他
--seed S             # 随机种子
--save FILE          # 保存结果到 JSON
--verbose            # 详细输出
```

### 示例用法

```bash
# 快速测试 (10 题, 5 轮)
python test_dp_gsm8k_scalable.py

# 标准测试 (100 题, 10 轮, 中等隐私)
python test_dp_gsm8k_scalable.py \
    --samples 100 \
    --epsilon 10.0 \
    --iterations 10 \
    --cache

# 大规模测试 (500 题, 强隐私)
python test_dp_gsm8k_scalable.py \
    --samples 500 \
    --epsilon 5.0 \
    --iterations 20 \
    --cache \
    --adaptive-clipping \
    --save results_500_eps5.json

# 低隐私高效用 (100 题, ε=20)
python test_dp_gsm8k_scalable.py \
    --samples 100 \
    --epsilon 20.0 \
    --iterations 15

# 使用基础组合 (更保守的预算估计)
python test_dp_gsm8k_scalable.py \
    --samples 50 \
    --composition basic
```

---

## 📈 推荐的测试流程

### Day 1: 验证和基础测试

```bash
# 1. 运行自动化测试套件
./run_all_tests.sh

# 2. 检查测试报告
cat test_report_*.txt

# 3. 快速 GSM8K 验证
python test_dp_gsm8k_scalable.py --samples 10
```

**预期时间**: 1 小时
**成功标准**: 所有单元测试通过, GSM8K 有改进

---

### Day 2: 中等规模实验

```bash
# 不同隐私预算对比
for eps in 2.0 5.0 10.0 20.0; do
    python test_dp_gsm8k_scalable.py \
        --samples 100 \
        --epsilon $eps \
        --iterations 10 \
        --save "results_100_eps${eps}.json"
done
```

**预期时间**: 4 小时
**分析**: 绘制 ε-准确率曲线

---

### Day 3: 大规模实验

```bash
# 500 题完整测试
python test_dp_gsm8k_scalable.py \
    --samples 500 \
    --epsilon 20.0 \
    --iterations 15 \
    --cache \
    --adaptive-clipping \
    --save results_500_full.json \
    > experiment_500.log 2>&1 &

# 监控进度
tail -f experiment_500.log
```

**预期时间**: 6-8 小时
**建议**: 后台运行 + 定期检查点

---

## 🔍 结果分析

### 查看 JSON 结果

```bash
# 美化输出
python -m json.tool results_100.json

# 提取关键指标
jq '.performance.improvement_percent' results_100.json
jq '.privacy.epsilon_consumed' results_100.json
```

### 对比多个实验

```python
import json
import matplotlib.pyplot as plt

# 加载结果
results = {}
for eps in [2.0, 5.0, 10.0, 20.0]:
    with open(f'results_100_eps{eps}.json') as f:
        results[eps] = json.load(f)

# 绘制隐私-效用曲线
epsilons = list(results.keys())
improvements = [r['performance']['improvement_percent'] for r in results.values()]

plt.plot(epsilons, improvements, marker='o')
plt.xlabel('Privacy Budget (ε)')
plt.ylabel('Improvement (%)')
plt.title('Privacy-Utility Tradeoff')
plt.savefig('privacy_utility_curve.png')
```

---

## ✅ 成功标准检查清单

### Phase 1: 修复验证
- [ ] `pytest tests/test_dp_es.py` 全部通过
- [ ] `test_phase2_simple.py` 显示正确的组合预算
- [ ] `test_debug_mode_fix.py` 确认无数据泄露

### Phase 2: 功能验证
- [ ] DP vs Non-DP 有明显效用差距 (>10%)
- [ ] 更小的 ε 导致更低的准确率
- [ ] 隐私预算正确消耗 (≤ target ε)

### Phase 3: 规模验证
- [ ] 100 题测试: 改进幅度 >15%
- [ ] 500 题测试: 改进幅度 >20%
- [ ] 隐私-效用曲线单调递增

---

## ⚠️  常见问题

### Q1: 测试失败: ModuleNotFoundError

```bash
# 确保在正确的目录
cd /Volumes/kk/dp-text-grad

# 确保依赖已安装
pip install -e .
```

### Q2: GSM8K 测试超时

```bash
# 减少样本数或迭代数
python test_dp_gsm8k_scalable.py --samples 5 --iterations 3
```

### Q3: 隐私预算超支

```bash
# 增加 target epsilon 或减少迭代
python test_dp_gsm8k_scalable.py --epsilon 10.0 --iterations 5
```

### Q4: 准确率没有改进

这是正常的! DP 可能降低效用:
- 尝试增加 ε (减少隐私保护)
- 增加种群大小 `--population 8`
- 启用缓存和自适应裁剪

---

## 📚 相关文档

- [完整测试路线图](TESTING_AND_SCALING_ROADMAP.md) - 详细计划
- [隐私修复总结](PRIVACY_FIXES_COMPLETE_SUMMARY.md) - Phase 1-4 修复
- [项目总览](PROJECT_PRIVACY_FIX_OVERVIEW.md) - 整体架构

---

## 🎯 下一步

完成基础测试后:

1. ✅ **验证修复** → 运行 `./run_all_tests.sh`
2. 📊 **小规模实验** → 10-100 题 GSM8K
3. 📈 **扩大规模** → 500-1000 题
4. 📑 **撰写论文** → 使用结果发表

**预计总时间**: 2-3 天完成所有测试和实验

---

**创建日期**: 2025-12-16
**最后更新**: 2025-12-16
