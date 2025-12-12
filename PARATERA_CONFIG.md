# Paratera 平台配置指南

## 平台信息

- **平台名称**: Paratera (并行科技)
- **API 地址**: `https://llmapi.paratera.com`
- **支持模型**: DeepSeek-V3.2-Exp 等多种国内大模型
- **余额**: 10,000 元 (仅可用于 DeepSeek 模型)

## 快速配置

### 方式 1: 使用 .env 文件(推荐)

在项目根目录的 `.env` 文件中配置:

```bash
# Paratera 平台配置
OPENAI_API_KEY=sk-Lyld88sT_oGZgcE9HyKoLg
OPENAI_API_BASE=https://llmapi.paratera.com
```

### 方式 2: 在代码中直接设置

```python
import os

# 设置 Paratera API
os.environ["OPENAI_API_KEY"] = "sk-Lyld88sT_oGZgcE9HyKoLg"
os.environ["OPENAI_API_BASE"] = "https://llmapi.paratera.com"
```

## 使用 DP-TextGrad

### 基本使用

```python
import dp_textgrad as tg

# 设置 DeepSeek 模型
tg.set_backward_engine("deepseek-chat", override=True)

# 或者使用其他 DeepSeek 模型
# tg.set_backward_engine("deepseek-v3", override=True)
# tg.set_backward_engine("DeepSeek-V3.2-Exp", override=True)
```

### 完整示例

```python
import os
import dp_textgrad as tg

# 配置 Paratera API
os.environ["OPENAI_API_KEY"] = "sk-Lyld88sT_oGZgcE9HyKoLg"
os.environ["OPENAI_API_BASE"] = "https://llmapi.paratera.com"

# 设置引擎
tg.set_backward_engine("deepseek-chat", override=True)

# 创建变量
prompt = tg.Variable(
    "写一个友好的问候语",
    role_description="任务指令",
    requires_grad=True
)

# 定义损失函数
loss_fn = tg.TextLoss("评估指令是否清晰友好，提供简短反馈")

# 计算损失
loss = loss_fn(prompt)

# 反向传播
loss.backward()

# 优化
optimizer = tg.TGD(parameters=[prompt])
optimizer.step()

print("优化后的提示:", prompt.get_value())
```

## 可用的 DeepSeek 模型

根据 Paratera 平台支持的模型,可能的模型名称包括:

1. **deepseek-chat** - 通用对话模型(推荐)
2. **deepseek-v3** - DeepSeek V3 模型
3. **DeepSeek-V3.2-Exp** - DeepSeek V3.2 实验版本
4. **deepseek-coder** - 代码专用模型

### 如何选择模型?

1. **运行测试脚本查询可用模型**:
   ```bash
   python test_paratera_simple.py
   ```

2. **测试不同模型**:
   ```python
   # 尝试不同的模型名称
   model_names = [
       "deepseek-chat",
       "deepseek-v3",
       "DeepSeek-V3.2-Exp",
       "deepseek-coder"
   ]

   for model in model_names:
       try:
           tg.set_backward_engine(model, override=True)
           print(f"✅ {model} 可用")
           break
       except:
           print(f"❌ {model} 不可用")
   ```

## 测试配置

### 1. 简单测试

```bash
python test_paratera_simple.py
```

这个脚本会:
- 查询 Paratera 平台的可用模型
- 测试选定的 DeepSeek 模型
- 验证 DP-TextGrad 集成

### 2. 完整测试

```bash
python test_paratera_api.py
```

这个脚本提供更详细的测试流程。

## 注意事项

### 1. 确保使用 DeepSeek 模型

你的 10,000 元余额**只能用于 DeepSeek 模型**。如果不小心使用了其他模型,可能会:
- 消耗其他渠道的额度
- 或者调用失败

### 2. 模型名称

不同平台对 DeepSeek 模型的命名可能不同:
- 官方 DeepSeek: `deepseek-chat`
- Paratera 平台: 可能是 `DeepSeek-V3.2-Exp` 或其他变体

**建议**: 先运行测试脚本查询确切的模型名称。

### 3. API 限流

虽然你有 10,000 元余额,但平台可能有请求频率限制。如果遇到限流错误:
- 在请求之间添加延迟
- 使用批处理减少请求次数

## 故障排查

### 问题 1: 模型名称错误

**错误**: `404` 或 `Model not found`

**解决方案**:
1. 运行 `python test_paratera_simple.py` 查看可用模型
2. 尝试这些常见名称:
   - `deepseek-chat`
   - `deepseek-v3`
   - `DeepSeek-V3.2-Exp`

### 问题 2: API Key 无效

**错误**: `401 Unauthorized`

**解决方案**:
1. 检查 API Key 是否正确
2. 确认 API Key 未过期
3. 确认账户余额充足

### 问题 3: 网络连接问题

**错误**: `timeout` 或 `connection error`

**解决方案**:
1. 检查网络连接
2. 如果需要代理,设置环境变量:
   ```bash
   export HTTP_PROXY=http://your-proxy:port
   export HTTPS_PROXY=http://your-proxy:port
   ```

## 成本估算

DeepSeek 模型通常比 GPT 便宜很多:

- **DeepSeek-V3**: 约 ¥0.001/1K tokens (输入)
- **10,000 元余额**: 可处理约 **1000 万 tokens**

这足够:
- 大量的提示优化实验
- 数百次完整的 DP-TextGrad 优化流程
- 数千次简单的 API 调用

## 进一步阅读

- [DP-TextGrad 快速开始](QUICKSTART.md)
- [DeepSeek 配置指南](DEEPSEEK_MINIMAX_CONFIG.md)
- [自定义 API 配置](CUSTOM_API_CONFIGURATION.md)

## 支持

如果遇到问题:

1. 首先运行测试脚本诊断
2. 查看错误信息和建议
3. 参考本文档的故障排查部分
4. 检查 Paratera 平台的官方文档
