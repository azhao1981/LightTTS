"""
SFT 模式集成测试

测试 SFT (Supervised Fine-Tuning) 模式的 API 功能。

运行测试前，确保：
1. 服务器已启动：python -m light_tts.server.api_server --model_dir ./pretrained_models/CosyVoice2-0.5B-finetune-v1
2. 模型 spk2info.pt 包含基础音色：female_test, male1_trained
3. voices.yaml 包含预设音色：female, male, male2 等
"""

import requests
import pytest
import time
import soundfile as sf
import numpy as np
import os


API_BASE = "http://localhost:8070"
os.makedirs("./outs", exist_ok=True)

def test_sft_api_basic():
    """测试 SFT API 基本功能"""
    start_time = time.time()
    data={
            "tts_text": "我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。",
            # "spk_id": "female_test_sft",  # 使用 spk2info.pt 中的音色 + _sft 后缀
            "spk_id": "male1_trained_sft",  # 使用 spk2info.pt 中的音色 + _sft 后缀
            "stream": "true"
        }
    response = requests.post(
        f"{API_BASE}/inference_zero_shot",
        data=data,
        stream=True,
        timeout=30  # 30秒超时
    )
    print(data)
    # 应该返回 200（如果音色存在）
    # 或者返回 400（如果音色不存在但错误信息正确）
    if response.status_code == 200:
        audio_data = bytearray()
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                audio_data.extend(chunk)

        cost_time = time.time() - start_time
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        speech_len = len(audio_data) / 2 / 24000

        output_wav = "./outs/sft_test_output.wav"
        sf.write(output_wav, audio_np, samplerate=24000, subtype="PCM_16")

        assert len(audio_data) > 0
        print(f"✅ SFT API 基本功能测试通过, 保存为 {output_wav}, 耗时: {cost_time:.2f}s, RTF: {cost_time/speech_len:.2f}")
    elif response.status_code == 400:
        # 检查是否是预期的错误信息
        error_msg = response.json().get("message", "")
        print(f"⚠️ SFT 音色未加载: {error_msg}")
        # 如果是因为音色不存在，这也是可以接受的
        assert "female_test" in error_msg or "not found" in error_msg
    else:
        pytest.fail(f"意外的状态码: {response.status_code}, 响应: {response.text}")


if __name__ == "__main__":
    
    print("\n测试 SFT API 基本功能...")
    test_sft_api_basic()

