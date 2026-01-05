import datetime
import os
import time

import numpy as np
import requests
import soundfile as sf
from dotenv import find_dotenv, load_dotenv

# 加载环境变量
load_dotenv(find_dotenv())

# 1. 配置服务地址
url = os.getenv("TTS_SERVICE_URL", "http://140.143.248.104:8090/inference_zero_shot")

# 2. 测试 speaker_id 功能（不上传音频文件）
# 注意：如果 preset 的 prompt_text 与音频内容不匹配，音色会很差
# 可以尝试传入空的 prompt_text，让系统使用 preset 的默认值
payload = {
    "tts_text": '你好，这是一个测试。',
    "speaker_id": "male",  # 使用 preset speaker
    "stream": False,  # 先用非流式测试
    # "prompt_text": ""  # 可选：如果不提供，会使用 preset 的 prompt_text
}

print(f"正在发送请求到 {url} ...")
print(f"使用 speaker_id: {payload['speaker_id']}")
print(f"流式模式: {payload['stream']}")

start_time = time.time()

# 3. 发送请求（不传 files）
try:
    with requests.post(url, data=payload, stream=payload['stream']) as response:
        response.raise_for_status()

        # 4. 接收音频数据
        output_filename = f"../asset/output_speaker_id_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        print("开始接收音频...")

        audio_data = bytearray()

        if payload['stream']:
            # 流式接收
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    audio_data.extend(chunk)
                    print(".", end="", flush=True)
        else:
            # 非流式，一次性接收
            audio_data = response.content
            print(f"接收到 {len(audio_data)} 字节")

    # 将字节数据转换为 NumPy 数组 (int16 格式)
    audio_np = np.frombuffer(audio_data, dtype=np.int16)

    # 使用 soundfile 保存为标准 WAV 格式 (采样率 24000)
    sample_rate = 24000
    sf.write(output_filename, audio_np, samplerate=sample_rate, subtype="PCM_16")

    # 计算音频时长和 RTF
    speech_len = len(audio_data) / 2 / sample_rate  # int16 每个样本 2 字节
    total_time = time.time() - start_time
    rtf = total_time / speech_len

    print(f"\n✅ 生成完成! 音频已保存为: {output_filename}")
    print(f"📊 音频时长: {speech_len:.2f} 秒")
    print(f"⏱️ 总耗时: {total_time:.2f} 秒")
    print(f"🚀 RTF (Real-Time Factor): {rtf:.2f}")

except requests.exceptions.ConnectionError:
    print("\n❌ 连接失败: 请检查服务是否在 8090 端口启动。")
except Exception as e:
    print(f"\n❌ 发生错误: {e}")
    import traceback
    traceback.print_exc()
