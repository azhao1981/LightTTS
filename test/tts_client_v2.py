import datetime
import os
import time

import numpy as np
import requests
import soundfile as sf
from dotenv import find_dotenv, load_dotenv

# 加载环境变量
load_dotenv(find_dotenv())

# 1. 配置服务地址 (注意端口 13099)
url = os.getenv("TTS_SERVICE_URL", "http://140.143.248.104:8090/inference_zero_shot")

# 2. 准备请求数据
# 请确保当前目录下有一个真实的 wav 文件
prompt_wav_path = "../assets/prompt_cut.mp3"

if not os.path.exists(prompt_wav_path):
    print(f"错误: 找不到参考音频 {prompt_wav_path}，请先准备一个wav文件。")
    exit()

payload = {
    "tts_text": '喂，李先生您好，我这边是先锋教育的王老师。打电话是想跟您同步一个对您家孩子可能有用的信息——我们这周末专门为初二学生安排了一场免费的物理试听课，重点就讲"浮力"这个最容易丢分的模块，帮孩子彻底搞懂原理和解题技巧。',
    "speaker_id": "male",
    "stream": True
}

files = {
    "prompt_wav": (os.path.basename(prompt_wav_path), open(prompt_wav_path, "rb"), "audio/mp3")
}

print(f"正在发送请求到 {url} ...")
start_time = time.time()

# 3. 发送流式请求 (关键是 stream=True)
try:
    with requests.post(url, data=payload, stream=True) as response:
        response.raise_for_status()

        # 4. 接收音频数据
        output_filename = f"../assets/output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        print("开始接收音频流...")

        # 记录首包到达时间 (TTFT)
        first_chunk_received = False

        # 使用 bytearray 收集音频数据
        audio_data = bytearray()

        # chunk_size=4096 分批读取
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                if not first_chunk_received:
                    ttft = time.time() - start_time
                    print(f"⚡ 首包延迟 (TTFT): {ttft:.3f} 秒")
                    first_chunk_received = True

                audio_data.extend(chunk)
                # 打印一个点表示收到一个数据包
                print(".", end="", flush=True)

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
    print("\n❌ 连接失败: 请检查服务是否在 8080 端口启动。")
except Exception as e:
    print(f"\n❌ 发生错误: {e}")
