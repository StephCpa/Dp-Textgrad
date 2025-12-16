# DeepSeek API 配置指南

## ⚠️ API Key 配置

当前 `.env` 文件中的 DeepSeek API key 显示为无效。你需要：

### 1. 获取有效的 DeepSeek API Key

访问 DeepSeek 官方平台获取 API key:
- **DeepSeek 官网**: https://platform.deepseek.com/
- **API 文档**: https://platform.deepseek.com/docs

### 2. 更新 `.env` 文件

编辑 `.env` 文件，替换为你的有效 API key:

```bash
# DeepSeek API配置
OPENAI_API_KEY=sk-your-valid-deepseek-key-here
OPENAI_API_BASE=https://api.deepseek.com/v1
```

### 3. 验证配置

运行测试脚本验证 API 配置:

```bash
python test_api_config.py
```

如果看到 "🎉 DeepSeek API 配置正确!"，说明配置成功。

---

## 🔄 替代方案

如果你没有 DeepSeek API key，可以使用其他兼容 OpenAI 格式的 API:

### 选项 1: OpenAI API

```bash
OPENAI_API_KEY=sk-your-openai-key
OPENAI_API_BASE=https://api.openai.com/v1  # 或留空
```

### 选项 2: 其他兼容 API（如智谱AI、Moonshot等）

```bash
OPENAI_API_KEY=your-key-here
OPENAI_API_BASE=https://your-provider-base-url
```

---

## 🚀 运行测试

配置好 API key 后，你可以：

### 快速验证 (不需要 API)
```bash
# 只运行不需要 API 的测试
./run_all_tests.sh
```

这会自动跳过需要 API 的 Phase 6（GSM8K 测试）。

### 完整测试 (需要 API)
```bash
# 配置好 API 后运行完整测试
./run_all_tests.sh
```

### 单独运行 GSM8K 测试
```bash
# 小规模测试 (10 题, 5-10 分钟)
python test_dp_gsm8k_scalable.py --samples 10 --epsilon 5.0

# 中等规模 (100 题, 1-2 小时)
python test_dp_gsm8k_scalable.py --samples 100 --epsilon 10.0 --cache
```

---

## 📊 当前测试状态

即使不配置 API，你仍然可以验证：

✅ **核心功能** - 所有 DP-ES 核心组件测试
✅ **隐私保护** - Phase 1-4 的所有隐私修复
✅ **统计审计** - DP 实现的正确性
✅ **单元测试** - 100% 通过率

**只有 Phase 6 (GSM8K 真实任务测试) 需要 API key**

---

## 💡 建议

### 如果你没有 API key：
- ✅ 继续使用当前的测试结果（10/10 通过）
- ✅ 所有核心功能都已验证
- ℹ️ Phase 6 (GSM8K) 可以跳过，不影响核心验证

### 如果你想运行完整测试：
1. 获取有效的 API key
2. 更新 `.env` 文件
3. 运行 `./run_all_tests.sh`

---

## ❓ 常见问题

**Q: 不配置 API 会影响测试吗？**
A: 不会。核心的 DP-ES 功能测试（Phase 1-5）都不需要 API。只有 Phase 6 (GSM8K真实任务测试) 需要 API。

**Q: 我应该用哪个 API 提供商？**
A: 推荐 DeepSeek（性价比高）或 OpenAI（稳定性好）。任何兼容 OpenAI 格式的 API 都可以使用。

**Q: 如何获取免费的 API key？**
A:
- DeepSeek: 新用户有免费额度
- OpenAI: 付费服务，但有 $5 免费额度（需信用卡）
- 部分国内提供商: 有免费试用

---

**更新日期**: 2025-12-16
