#!/bin/bash
# SFT 模式端到端测试脚本

set -e

echo "=========================================="
echo "SFT 模式端到端测试"
echo "=========================================="

API_BASE="http://localhost:8080"
OUTPUT_DIR="test_output/sft"
mkdir -p "$OUTPUT_DIR"

# 测试 1: Zero-Shot 模式（现有功能）
echo ""
echo "[测试 1] Zero-Shot 模式（现有功能）..."
echo "---------------------------------------"
response=$(curl -s -X POST "$API_BASE/inference_zero_shot" \
  -F "tts_text=你好，这是 Zero-Shot 模式的语音合成测试。" \
  -F "spk_id=female_test" \
  -F "stream=false" \
  -o "$OUTPUT_DIR/test_zero_shot.wav" \
  -w "%{http_code}")

if [ "$response" = "200" ]; then
    echo "✅ Zero-Shot 模式测试通过"
    echo "   输出: $OUTPUT_DIR/test_zero_shot.wav"
    ls -lh "$OUTPUT_DIR/test_zero_shot.wav"
else
    echo "❌ Zero-Shot 模式测试失败 (HTTP $response)"
fi

# 测试 2: SFT 模式（新功能）
echo ""
echo "[测试 2] SFT 模式（新功能）..."
echo "---------------------------------------"
response=$(curl -s -X POST "$API_BASE/inference_zero_shot" \
  -F "tts_text=你好，这是 SFT 模式的语音合成测试。" \
  -F "spk_id=female_test_sft" \
  -F "stream=false" \
  -o "$OUTPUT_DIR/test_sft.wav" \
  -w "%{http_code}")

if [ "$response" = "200" ]; then
    echo "✅ SFT 模式测试通过"
    echo "   输出: $OUTPUT_DIR/test_sft.wav"
    ls -lh "$OUTPUT_DIR/test_sft.wav"
else
    echo "⚠️ SFT 模式测试失败 (HTTP $response)"
    echo "   可能原因: 模型中未加载 SFT 音色或 LLM 不支持 semantic_len=0"
fi

# 测试 3: 错误处理 - 无效 spk_id
echo ""
echo "[测试 3] 错误处理 - 无效 spk_id..."
echo "---------------------------------------"
response=$(curl -s -X POST "$API_BASE/inference_zero_shot" \
  -F "tts_text=测试" \
  -F "spk_id=invalid_sft")

if echo "$response" | grep -q "Invalid\|not found"; then
    echo "✅ 错误处理正确"
    echo "   错误信息: $response"
else
    echo "❌ 错误处理失败"
    echo "   响应: $response"
fi

# 测试 4: 对比两种模式
echo ""
echo "[测试 4] 对比 Zero-Shot 和 SFT 模式..."
echo "---------------------------------------"
text="同样的文本用于对比测试。"

echo "生成 Zero-Shot 模式音频..."
curl -s -X POST "$API_BASE/inference_zero_shot" \
  -F "tts_text=$text" \
  -F "spk_id=female_test" \
  -o "$OUTPUT_DIR/compare_zero_shot.wav"

echo "生成 SFT 模式音频..."
curl -s -X POST "$API_BASE/inference_zero_shot" \
  -F "tts_text=$text" \
  -F "spk_id=female_test_sft" \
  -o "$OUTPUT_DIR/compare_sft.wav"

echo ""
echo "✅ 对比文件已生成:"
ls -lh "$OUTPUT_DIR/compare_*.wav

# 总结
echo ""
echo "=========================================="
echo "测试总结"
echo "=========================================="
echo "输出目录: $OUTPUT_DIR"
echo ""
echo "文件列表:"
ls -lh "$OUTPUT_DIR" || echo "（无输出文件）"
echo ""
echo "=========================================="
echo "所有测试完成！"
echo "=========================================="
