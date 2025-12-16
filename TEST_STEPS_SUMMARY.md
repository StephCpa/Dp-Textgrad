# DP-TextGrad 测试步骤总结

**日期**: 2025-12-16
**当前状态**: Phase 1-4 修复完成 ✅，准备系统测试

---

## 🎯 测试步骤概览

### 步骤 1: 基础验证 (5 分钟) ⚡

**目的**: 确认所有修复正确工作

```bash
# 运行核心单元测试
python -m pytest tests/test_dp_es.py -v
```

**预期结果**: 6 个测试全部通过 ✅

**验证点**:
- ✅ DPScorer 正确实现组合定理
- ✅ CritiqueOption 支持哈希
- ✅ 隐私预算计算正确
- ✅ 高级组合计算正确

---

### 步骤 2: Phase 修复验证 (10 分钟)

**目的**: 确认 Phase 2-4 的关键修复

```bash
# Phase 2: 隐私预算修复
python test_phase2_simple.py

# Phase 4: DPScoreRecord 修复
python test_debug_mode_fix.py
```

**预期结果**:

**Phase 2**:
```
候选数量: 4
每个候选: ε=0.5
总预算: ε=2.0000 ✅ (基础组合)

反馈默认禁用: ✅ PASS
反馈警告显示: ✅ PASS
```

**Phase 4**:
```
TEST 1: Production mode protects privacy
  ✅ PASS: Records are empty in production mode

TEST 2: Debug mode works with warning
  ✅ PASS: Debug mode works correctly

TEST 3: Production code cannot leak
  ✅ PROTECTED: records list is empty

TEST 4: Normal operations still work
  ✅ PASS: All normal DP operations work correctly
```

---

### 步骤 3: 严格审计 (5 分钟)

**目的**: 验证 DP 实现的统计正确性

```bash
python test_strict_audit.py
```

**预期结果**:
```
TEST 1: Noise Distribution Audit
  K-S test: p-value = 0.92
  ✅ PASS: Gaussian noise passes

TEST 2: Privacy Loss Distribution Analysis
  Pr[loss > ε]: 0.032
  95% CI: [0.023, 0.045]
  ✅ PASS: Satisfies (ε, δ)-DP
```

---

### 步骤 4: 快速功能测试 (5 分钟)

**目的**: 验证 DP-ES 端到端工作

```bash
# 可选: 最小工作示例
python test_dp_minimal_working.py
```

**预期**: 能完成优化流程，隐私预算正确消耗

---

### 步骤 5: 小规模实验 (5 分钟) 🚀

**目的**: 在真实场景下快速验证

```bash
# GSM8K 10 题快速测试
python test_dp_gsm8k_scalable.py --samples 10 --epsilon 5.0
```

**预期输出**:
```
[5/6] 运行 DP-ES 优化 (5 轮)...
  初始分数: 0.350

  轮次 1/5:
    当前分数: 0.420
    隐私预算: 已用 0.45, 剩余 4.55

  ...

  最终分数: 0.580
  改进幅度: +0.230 (+65.7%)

实验摘要
样本数量: 10
初始分数: 0.350
最终分数: 0.580
改进幅度: +0.230 (+65.7%)
隐私消耗: ε=2.250 (目标: 5.0)
运行时间: 28.3 秒

✅ 实验成功: 提示有改进!
```

---

## 📋 完整测试清单

### 必须通过的测试 (核心验证)

- [ ] `pytest tests/test_dp_es.py -v` → 6 passed ✅
- [ ] `python test_phase2_simple.py` → 所有检查通过 ✅
- [ ] `python test_debug_mode_fix.py` → 4 个测试通过 ✅
- [ ] `python test_strict_audit.py` → 噪声和隐私损失审计通过 ✅

### 功能测试 (可选但推荐)

- [ ] `python test_dp_minimal_working.py` → 完成优化 ✅
- [ ] `python test_dp_gsm8k_scalable.py --samples 10` → 有改进 ✅

### 集成测试 (如果有 API key)

- [ ] `python -m pytest tests/test_api.py -v` → 需要 OPENAI_API_KEY

---

## 🚀 一键运行所有测试

```bash
# 自动化测试脚本 (推荐)
./run_all_tests.sh
```

这会:
1. 运行所有核心测试
2. 跳过需要 API key 的测试
3. 生成测试报告
4. 显示通过/失败统计

**预期输出**:
```
================================
[Phase 1/6] 核心单元测试
================================

Running: DP-ES 核心组件测试
✅ DP-ES 核心组件测试 PASSED

...

================================
测试结果总结
================================
总测试数: 8+
通过: 8+
失败: 0

✅ 测试报告已保存到: test_report_20251216_143022.txt

================================
🎉 所有测试通过!
================================
```

---

## 📊 扩大实验规模 (可选)

完成基础验证后，可以扩大实验规模:

### 中等规模 (100 题, 2 小时)

```bash
python test_dp_gsm8k_scalable.py \
    --samples 100 \
    --epsilon 10.0 \
    --iterations 10 \
    --cache \
    --save results_100.json
```

### 大规模 (500 题, 6 小时)

```bash
nohup python test_dp_gsm8k_scalable.py \
    --samples 500 \
    --epsilon 20.0 \
    --iterations 15 \
    --cache \
    --adaptive-clipping \
    --save results_500.json \
    > experiment_500.log 2>&1 &

# 监控进度
tail -f experiment_500.log
```

### 隐私-效用权衡分析

```bash
# 测试不同 ε 值
for eps in 2.0 5.0 10.0 20.0; do
    python test_dp_gsm8k_scalable.py \
        --samples 100 \
        --epsilon $eps \
        --iterations 10 \
        --save "results_100_eps${eps}.json"
done

# 然后分析结果绘制曲线
```

---

## ⏱️ 预计时间

| 测试阶段 | 耗时 | 说明 |
|---------|------|------|
| 步骤 1-4 | 25 分钟 | 核心验证 |
| 步骤 5 | 5 分钟 | 快速功能测试 |
| **总计 (基础)** | **30 分钟** | **足够验证修复正确** |
| 中等规模 | 2 小时 | 100 题实验 (可选) |
| 大规模 | 6-8 小时 | 500+ 题 (可选) |

---

## ✅ 成功标准

### 基础验证通过标准

1. ✅ `tests/test_dp_es.py`: 6/6 通过
2. ✅ `test_phase2_simple.py`: 显示 "✅ PASS" 4 次
3. ✅ `test_debug_mode_fix.py`: 显示 "🎉 ALL TESTS PASSED!"
4. ✅ `test_strict_audit.py`: K-S 检验和隐私损失检验通过

### 功能验证通过标准

5. ✅ GSM8K 小规模: 改进幅度 > 0 (任何正改进都算成功)
6. ✅ 隐私预算: consumed_epsilon ≤ target_epsilon

---

## 🐛 常见问题

### Q: pytest 找不到模块?

```bash
# 确保在项目根目录
cd /Volumes/kk/dp-text-grad

# 安装项目 (可编辑模式)
pip install -e .
```

### Q: test_api.py 失败?

这是正常的! 需要 OPENAI_API_KEY。如果没有 API key:
- 测试脚本会自动跳过
- 或手动跳过: `pytest tests/test_dp_es.py tests/test_basics.py -v`

### Q: 改进幅度为 0 或负数?

这可能发生在:
- ε 太小 (隐私保护太强)
- 样本数太少 (运气不好)
- 迭代数太少

**解决方法**: 增加 `--epsilon` 或 `--iterations`

---

## 📚 相关文档

- **详细路线图**: [TESTING_AND_SCALING_ROADMAP.md](TESTING_AND_SCALING_ROADMAP.md)
- **快速指南**: [QUICK_START_TESTING.md](QUICK_START_TESTING.md)
- **修复总结**: [PRIVACY_FIXES_COMPLETE_SUMMARY.md](PRIVACY_FIXES_COMPLETE_SUMMARY.md)

---

## 🎯 立即开始

**最简单的方式 (推荐新手)**:

```bash
# 一条命令完成所有基础测试
./run_all_tests.sh
```

**手动逐步验证 (如果想看细节)**:

```bash
# 1. 核心测试
python -m pytest tests/test_dp_es.py -v

# 2. Phase 验证
python test_phase2_simple.py
python test_debug_mode_fix.py

# 3. 审计
python test_strict_audit.py

# 4. 快速实验
python test_dp_gsm8k_scalable.py --samples 10
```

---

**更新时间**: 2025-12-16
**总预计时间**: 30 分钟 (基础验证)
**成功率**: 预期 100% 通过 ✅
