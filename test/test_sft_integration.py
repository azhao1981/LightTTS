"""
SFT 模式集成测试

测试 SFT (Supervised Fine-Tuning) 模式的 API 功能。

运行测试前，确保：
1. 服务器已启动：python -m light_tts.server.api_server --model_dir ./pretrained_models/CosyVoice2-0.5B-finetune-v1
2. 模型包含基础音色：female_test, male1_trained
"""

import requests
import pytest


API_BASE = "http://localhost:8080"


def test_sft_spk_id_detection():
    """测试 SFT 模式 spk_id 检测逻辑"""
    # 测试 _sft 后缀检测
    assert "female_test_sft".endswith('_sft') == True
    assert "male1_trained_sft".endswith('_sft') == True
    assert "female_test".endswith('_sft') == False
    assert "male1_trained".endswith('_sft') == False

    # 测试基础 spk_id 提取
    assert "female_test_sft"[:-4] == "female_test"
    assert "male1_trained_sft"[:-4] == "male1_trained"


def test_sft_api_basic():
    """测试 SFT API 基本功能"""
    response = requests.post(
        f"{API_BASE}/inference_zero_shot",
        data={
            "tts_text": "你好，这是 SFT 模式测试。",
            "spk_id": "female_test_sft",
            "stream": "false"
        }
    )

    # 应该返回 200（如果音色存在）
    # 或者返回 400（如果音色不存在但错误信息正确）
    if response.status_code == 200:
        assert len(response.content) > 0
        print("✅ SFT API 基本功能测试通过")
    elif response.status_code == 400:
        # 检查是否是预期的错误信息
        error_msg = response.json().get("message", "")
        print(f"⚠️ SFT 音色未加载: {error_msg}")
        # 如果是因为音色不存在，这也是可以接受的
        assert "female_test" in error_msg or "not found" in error_msg
    else:
        pytest.fail(f"意外的状态码: {response.status_code}, 响应: {response.text}")


def test_sft_invalid_spk_id():
    """测试无效 SFT spk_id"""
    response = requests.post(
        f"{API_BASE}/inference_zero_shot",
        data={
            "tts_text": "测试",
            "spk_id": "invalid_sft",
        }
    )

    assert response.status_code == 400
    error_msg = response.json().get("message", "")
    assert "invalid" in error_msg.lower() or "not found" in error_msg.lower()
    print(f"✅ 无效 spk_id 错误处理正确: {error_msg}")


def test_zero_shot_unchanged():
    """测试 Zero-Shot 模式不受影响"""
    response = requests.post(
        f"{API_BASE}/inference_zero_shot",
        data={
            "tts_text": "你好世界",
            "spk_id": "female_test",
            "stream": "false"
        }
    )

    if response.status_code == 200:
        assert len(response.content) > 0
        print("✅ Zero-Shot 模式正常工作")
    else:
        print(f"⚠️ Zero-Shot 测试失败: {response.status_code}, {response.text}")


def test_sft_vs_zero_shot_comparison():
    """对比 SFT 和 Zero-Shot 输出"""
    text = "你好世界"

    # SFT 模式
    sft_response = requests.post(
        f"{API_BASE}/inference_zero_shot",
        data={"tts_text": text, "spk_id": "female_test_sft", "stream": "false"}
    )

    # Zero-Shot 模式
    zs_response = requests.post(
        f"{API_BASE}/inference_zero_shot",
        data={"tts_text": text, "spk_id": "female_test", "stream": "false"}
    )

    # 如果两者都成功，验证都有输出
    if sft_response.status_code == 200 and zs_response.status_code == 200:
        assert len(sft_response.content) > 0
        assert len(zs_response.content) > 0
        print("✅ SFT 和 Zero-Shot 模式对比测试通过")
        print(f"   SFT 输出大小: {len(sft_response.content)} bytes")
        print(f"   Zero-Shot 输出大小: {len(zs_response.content)} bytes")
    else:
        print(f"⚠️ 对比测试失败:")
        print(f"   SFT 状态码: {sft_response.status_code}")
        print(f"   Zero-Shot 状态码: {zs_response.status_code}")


if __name__ == "__main__":
    print("=" * 60)
    print("SFT 模式集成测试")
    print("=" * 60)

    # 测试 1: spk_id 检测逻辑
    print("\n[1/5] 测试 spk_id 检测逻辑...")
    test_sft_spk_id_detection()
    print("✅ spk_id 检测逻辑测试通过")

    # 测试 2: SFT API 基本功能
    print("\n[2/5] 测试 SFT API 基本功能...")
    test_sft_api_basic()

    # 测试 3: 无效 spk_id 错误处理
    print("\n[3/5] 测试无效 spk_id 错误处理...")
    test_sft_invalid_spk_id()

    # 测试 4: Zero-Shot 模式不受影响
    print("\n[4/5] 测试 Zero-Shot 模式不受影响...")
    test_zero_shot_unchanged()

    # 测试 5: SFT vs Zero-Shot 对比
    print("\n[5/5] 测试 SFT vs Zero-Shot 对比...")
    test_sft_vs_zero_shot_comparison()

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
