#!/usr/bin/env python3
"""
测试预设音色功能

Usage:
    python test/test_presets.py --text "你好,世界!" --spk_id female --output test_output.wav
"""

import requests
import argparse
import soundfile as sf
import numpy as np


def query_presets(base_url="http://localhost:8090"):
    """查询可用的预设音色"""
    response = requests.get(f"{base_url}/query_presets")
    if response.status_code == 200:
        data = response.json()
        print("可用预设音色:")
        print(f"  总数: {data['total_count']}")
        print(f"  成功加载: {data['loaded_count']}")
        print(f"  加载失败: {data['failed_count']}")
        print(f"  音色列表: {data['presets']}")
        return data['presets']
    else:
        print(f"查询失败: {response.status_code}")
        return []


def test_preset_inference(text, spk_id, output_file, base_url="http://localhost:8090", stream=False):
    """测试使用预设音色进行推理"""
    print(f"\n{'='*60}")
    print(f"测试预设音色推理")
    print(f"{'='*60}")
    print(f"文本: {text}")
    print(f"音色 ID: {spk_id}")
    print(f"流式: {stream}")
    print(f"{'='*60}\n")

    data = {
        "tts_text": text,
        "spk_id": spk_id,
        "stream": stream,
    }

    response = requests.post(f"{base_url}/inference_zero_shot", data=data, stream=True)

    if response.status_code == 200:
        audio_data = bytearray()
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                audio_data.extend(chunk)

        # 转换为 numpy 数组
        audio_np = np.frombuffer(audio_data, dtype=np.int16)

        # 保存音频文件
        sf.write(output_file, audio_np, samplerate=24000, subtype="PCM_16")
        print(f"✓ 成功生成音频: {output_file}")
        print(f"  音频时长: {len(audio_np) / 24000:.2f} 秒")
        print(f"  文件大小: {len(audio_data)} 字节")
        return True
    else:
        print(f"✗ 推理失败: {response.status_code}")
        print(f"  错误信息: {response.text}")
        return False


# python test/test_presets.py   --text "测试音色质量"   --spk_id male_long   --output test_male_long.wav   --url http://140.143.248.104:8090 --stream
def main():
    parser = argparse.ArgumentParser(description="测试预设音色功能")
    parser.add_argument("--text", type=str, default="你好,这是一个测试。", help="要合成的文本")
    parser.add_argument("--spk_id", type=str, default="female", help="预设音色 ID")
    parser.add_argument("--output", type=str, default="test_output.wav", help="输出音频文件")
    parser.add_argument("--url", type=str, default="http://localhost:8090", help="服务器地址")
    parser.add_argument("--stream", action="store_true", help="使用流式推理")
    parser.add_argument("--query", action="store_true", help="仅查询可用音色")

    args = parser.parse_args()

    # 查询可用音色
    presets = query_presets(args.url)

    if args.query:
        return

    # 检查音色 ID 是否有效
    if args.spk_id not in presets:
        print(f"✗ 无效的音色 ID: {args.spk_id}")
        print(f"  可用音色: {presets}")
        return

    # 测试推理
    success = test_preset_inference(
        text=args.text,
        spk_id=args.spk_id,
        output_file=args.output,
        base_url=args.url,
        stream=args.stream
    )

    if success:
        print("\n✓ 测试完成!")
    else:
        print("\n✗ 测试失败!")
        exit(1)


if __name__ == "__main__":
    main()
