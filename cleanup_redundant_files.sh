#!/bin/bash
# DP-TextGrad 冗余文件清理脚本
# 创建日期: 2025-12-16

echo "========================================================================"
echo "DP-TextGrad 项目文件清理"
echo "========================================================================"
echo ""

# 创建归档目录
mkdir -p archive/{debug_tests,phase_docs,old_configs,test_reports,old_tests}

# 1. 移动调试测试文件
echo "[1/5] 移动调试测试文件..."
mv debug_eval.py archive/debug_tests/ 2>/dev/null
mv debug_eval2.py archive/debug_tests/ 2>/dev/null
mv debug_gsm8k.py archive/debug_tests/ 2>/dev/null
mv test_parse.py archive/debug_tests/ 2>/dev/null
mv test_fix.py archive/debug_tests/ 2>/dev/null
echo "✓ 完成"

# 2. 移动Phase相关文档（已被TEST_FIXES_FINAL_SUMMARY.md取代）
echo "[2/5] 移动Phase相关文档..."
mv PHASE2_COMPLETION_SUMMARY.md archive/phase_docs/ 2>/dev/null
mv PHASE2_PRIVACY_FIX_PLAN.md archive/phase_docs/ 2>/dev/null
mv PHASE3_DP_AUDIT_PLAN.md archive/phase_docs/ 2>/dev/null
mv PHASE4_DPSCORERECORD_FIX_SUMMARY.md archive/phase_docs/ 2>/dev/null
mv PRIVACY_FIX_PLAN.md archive/phase_docs/ 2>/dev/null
mv PRIVACY_FIX_PROGRESS.md archive/phase_docs/ 2>/dev/null
mv PRIVACY_FIX_SUMMARY.md archive/phase_docs/ 2>/dev/null
mv PRIVACY_FIXES_COMPLETE_SUMMARY.md archive/phase_docs/ 2>/dev/null
mv PROJECT_PRIVACY_FIX_OVERVIEW.md archive/phase_docs/ 2>/dev/null
echo "✓ 完成"

# 3. 移动旧配置和分析文档
echo "[3/5] 移动旧配置文档..."
mv PARATERA_CONFIG.md archive/old_configs/ 2>/dev/null
mv OPTIMIZATION_ANALYSIS_REPORT.md archive/old_configs/ 2>/dev/null
mv OPTIMIZATION_SUMMARY.md archive/old_configs/ 2>/dev/null
mv EXPERIMENT_REPORT.md archive/old_configs/ 2>/dev/null
echo "✓ 完成"

# 4. 移动测试报告
echo "[4/5] 移动测试报告..."
mv test_report_*.txt archive/test_reports/ 2>/dev/null
echo "✓ 完成"

# 5. 移动旧版本测试文件（保留v3和scalable版本）
echo "[5/5] 移动旧版本测试文件..."
mv test_dp_gsm8k.py archive/old_tests/ 2>/dev/null
mv test_dp_gsm8k_v2.py archive/old_tests/ 2>/dev/null
mv test_dp_improved.py archive/old_tests/ 2>/dev/null
mv test_dp_comprehensive.py archive/old_tests/ 2>/dev/null
mv test_privacy_utility_curve.py archive/old_tests/ 2>/dev/null
echo "✓ 完成"

# 统计
echo ""
echo "========================================================================"
echo "清理完成统计"
echo "========================================================================"
echo "归档目录结构:"
echo ""
find archive -type f | sort
echo ""
echo "剩余顶层测试文件:"
ls -1 test_*.py 2>/dev/null | wc -l
echo ""
echo "剩余顶层文档文件:"
ls -1 *.md 2>/dev/null | wc -l
echo ""
echo "========================================================================"
echo "✅ 清理完成！所有冗余文件已移至 archive/ 目录"
echo "========================================================================"
