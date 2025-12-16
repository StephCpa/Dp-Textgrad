#!/usr/bin/env python3
"""
快速测试 DeepSeek API 配置
"""
import os
import sys

# 直接设置环境变量（不依赖 dotenv）
os.environ['OPENAI_API_KEY'] = 'sk-ad2c241ee4924e9483cb2059024eda91'
os.environ['OPENAI_API_BASE'] = 'https://api.deepseek.com/v1'

print("=" * 80)
print(" DeepSeek API 配置测试")
print("=" * 80)

# 检查环境变量
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")

print(f"\n✓ OPENAI_API_KEY: {api_key[:20]}...{api_key[-10:] if api_key else 'NOT SET'}")
print(f"✓ OPENAI_API_BASE: {api_base if api_base else 'NOT SET'}")

if not api_key:
    print("\n❌ 错误: OPENAI_API_KEY 未设置")
    sys.exit(1)

# 测试 API 连接
print("\n" + "=" * 80)
print(" 测试 API 连接")
print("=" * 80)

try:
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=api_base
    )

    print(f"\n尝试连接到: {api_base}")
    print(f"使用模型: deepseek-chat")

    # 简单测试
    response = client.chat.completions.create(
        model='deepseek-chat',
        messages=[{'role': 'user', 'content': '1+1等于几？只回答数字'}],
        max_tokens=10
    )

    answer = response.choices[0].message.content

    print(f"\n✅ API 连接成功!")
    print(f"测试响应: {answer}")

    print("\n" + "=" * 80)
    print(" 🎉 DeepSeek API 配置正确!")
    print("=" * 80)
    print("\n你可以运行完整测试了:")
    print("  export OPENAI_API_KEY='sk-ad2c241ee4924e9483cb2059024eda91'")
    print("  export OPENAI_API_BASE='https://api.deepseek.com/v1'")
    print("  ./run_all_tests.sh")
    print("\n或单独运行 GSM8K 测试:")
    print("  python test_dp_gsm8k_scalable.py --samples 10")

except Exception as e:
    print(f"\n❌ API 连接失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
