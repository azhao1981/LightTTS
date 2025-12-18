#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightTTS CosyVoice2 生产部署验证脚本 (可直接执行)

功能：
1. 健康检查
2. 零样本TTS推理测试
3. 性能基准测试
4. 并发压力测试

依赖: 会自动尝试导入或提示安装
"""

import sys
import os

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 环境初始化 (自动尝试激活)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def setup_environment():
    """自动处理环境激活"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir
    parent_dir = os.path.dirname(project_root)

    # 尝试在 sys.path 中添加可能的 venv site-packages
    venv_candidates = [
        os.path.join(parent_dir, "tts-research", ".venv", "lib", "python3.11", "site-packages"),
        os.path.join(parent_dir, "tts-research", ".venv", "lib", "python3.10", "site-packages"),
        os.path.join(project_root, ".venv", "lib", "python3.11", "site-packages"),
        os.path.join(parent_dir, ".venv", "lib", "python3.11", "site-packages"),
    ]

    for venv_path in venv_candidates:
        if os.path.exists(venv_path) and venv_path not in sys.path:
            sys.path.insert(0, venv_path)
            print(f"✓ 自动添加路径: {venv_path}")

# 尝试自动化环境设置
setup_environment()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 依赖检查
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_and_import(package_name, pip_name=None):
    """检查并导入包，如果失败则提示安装"""
    if pip_name is None:
        pip_name = package_name
    try:
        __import__(package_name)
        return True
    except ImportError:
        print(f"错误: 缺少依赖包 '{package_name}'")
        print(f"  请执行: pip install {pip_name}")
        return False

# 检查核心依赖
missing_deps = []
for pkg, pip_pkg in [
    ("requests", "requests"),
    ("soundfile", "soundfile"),
    ("numpy", "numpy"),
]:
    if not check_and_import(pkg, pip_pkg):
        missing_deps.append(pip_pkg)

if missing_deps:
    print(f"\n缺少依赖: {', '.join(missing_deps)}")
    print("正在尝试自动安装...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_deps)
    print("安装完成！\n")

# 现在导入
import json
import time
import wave
from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
import requests
import soundfile as sf

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT_SERVER = "http://localhost:8080"
DEFAULT_TIMEOUT = 60

# 测试数据
TEST_PROMPT_WAV = "/root/tts/light-tts/asset/demo_user.wav"
TEST_PROMPT_TEXT = "你好，我是测试用户。"
TEST_TTS_TEXT = "你好，我是LightTTS测试。这是一个生产部署验证的语音合成演示。"

def generate_test_wav_if_missing():
    """生成静音测试音频"""
    if not os.path.exists(TEST_PROMPT_WAV):
        print(f"生成测试音频: {TEST_PROMPT_WAV}")
        os.makedirs(os.path.dirname(TEST_PROMPT_WAV), exist_ok=True)
        sample_rate = 16000
        duration = 1.5
        silence = np.zeros(int(sample_rate * duration), dtype=np.int16)
        with wave.open(TEST_PROMPT_WAV, 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sample_rate)
            f.writeframes(silence.tobytes())

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 核心测试函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_health(server: str) -> Tuple[bool, Dict]:
    """健康检查"""
    try:
        response = requests.get(f"{server}/healthz", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else {"error": response.status_code}
    except Exception as e:
        return False, {"error": str(e)}

def query_models(server: str) -> Tuple[bool, List]:
    """查询模型"""
    try:
        response = requests.get(f"{server}/query_tts_model", timeout=5)
        data = response.json()
        return True, data.get("tts_models", [])
    except:
        return False, []

def inference_zero_shot(server: str, stream: bool = False, timeout: int = 30) -> Tuple[bool, float, bytes]:
    """流式/非流式推理测试"""
    generate_test_wav_if_missing()
    if not os.path.exists(TEST_PROMPT_WAV):
        return False, 0, b""

    files = {'prompt_wav': ('prompt.wav', open(TEST_PROMPT_WAV, 'rb'), 'audio/wav')}
    data = {
        'tts_text': TEST_TTS_TEXT,
        'prompt_text': TEST_PROMPT_TEXT,
        'stream': str(stream).lower(),
        'tts_model_name': 'default',
        'speed': 1.0,
    }

    try:
        start = time.time()
        response = requests.post(
            f"{server}/inference_zero_shot",
            files=files,
            data=data,
            timeout=timeout,
            stream=stream
        )
        latency = (time.time() - start) * 1000

        if response.status_code == 200:
            return True, latency, response.content
        return False, latency, b""
    except requests.exceptions.Timeout:
        return False, timeout * 1000, b""
    except Exception as e:
        return False, 0, b"".encode(str(e), "utf-8")

def save_audio(audio_bytes: bytes, path: str):
    """保存音频"""
    try:
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        sf.write(path, audio_array, 24000)
        print(f"  保存音频: {path}")
    except Exception as e:
        print(f"  保存失败: {e}")

def benchmark_concurrent(server: str, num_requests: int = 3, concurrent: int = 2) -> Dict:
    """并发性能测试"""
    import concurrent.futures
    import threading

    results = {'total': num_requests, 'success': 0, 'failed': 0, 'latencies': [], 'errors': []}
    lock = threading.Lock()

    def single_test(idx):
        success, latency, _ = inference_zero_shot(server, stream=False, timeout=45)
        with lock:
            if success:
                results['latencies'].append(latency)
            else:
                results['errors'].append(idx)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = [executor.submit(single_test, i) for i in range(num_requests)]
        concurrent.futures.wait(futures)

    results['success'] = len(results['latencies'])
    results['failed'] = len(results['errors'])

    if results['latencies']:
        results['avg_ms'] = np.mean(results['latencies'])
        results['p95_ms'] = np.percentile(results['latencies'], 95)
        results['min_ms'] = min(results['latencies'])
        results['max_ms'] = max(results['latencies'])

    return results

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主测试套件
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_tests(server: str, output_dir: str, full: bool = False):
    """执行完整测试"""
    os.makedirs(output_dir, exist_ok=True)
    results = {'server': server, 'timestamp': time.time(), 'tests': {}, 'verdict': 'PENDING'}

    print("=" * 70)
    print("LightTTS CosyVoice2 生产部署验证")
    print("=" * 70)
    print(f"服务器: {server}")
    print(f"输出目录: {output_dir}")
    print(f"完整测试: {full}")
    print("=" * 70)

    # 1. 健康检查
    print("\n[1/5] 健康检查...")
    is_healthy, health = check_health(server)
    results['tests']['health'] = {'pass': is_healthy, 'data': health}
    print(f"  {'✓ 通过' if is_healthy else '✗ 失败'}: {health}")
    if not is_healthy:
        results['verdict'] = 'FAILED'
        return results

    # 2. 模型查询
    print("\n[2/5] 模型查询...")
    success, models = query_models(server)
    results['tests']['models'] = {'pass': success, 'models': models}
    print(f"  {'✓' if success else '✗'} 可用模型: {models}")

    # 3. 推理测试
    print("\n[3/5] 推理测试 (非流式)...")
    success, latency, audio = inference_zero_shot(server, stream=False, timeout=30)
    results['tests']['inference'] = {'pass': success, 'latency_ms': latency}
    if success:
        print(f"  ✓ 成功: {latency:.0f}ms")
        save_audio(audio, os.path.join(output_dir, "inference.wav"))
    else:
        print(f"  ✗ 失败")

    # 4. 性能测试
    if full:
        print("\n[4/5] 并发性能测试 (3请求, 2并发)...")
        perf = benchmark_concurrent(server, 3, 2)
        results['tests']['performance'] = perf
        if perf['failed'] == 0:
            print(f"  ✓ 100%成功, 平均: {perf['avg_ms']:.0f}ms, P95: {perf['p95_ms']:.0f}ms")
        else:
            print(f"  ⚠ 成功 {perf['success']}/3")

    # 5. 流式测试
    if full:
        print("\n[5/5] 流式推理...")
        stream_success, stream_latency, stream_audio = inference_zero_shot(server, stream=True, timeout=45)
        results['tests']['streaming'] = {'pass': stream_success, 'latency_ms': stream_latency}
        if stream_success:
            print(f"  ✓ 流式成功: {stream_latency:.0f}ms")
            save_audio(stream_audio, os.path.join(output_dir, "streaming.wav"))
        else:
            print(f"  ✗ 流式失败")

    # 判决
    base_passed = (results['tests']['health']['pass'] and
                   results['tests']['models']['pass'] and
                   results['tests']['inference']['pass'])

    if full:
        all_passed = base_passed and results['tests']['performance']['failed'] == 0 and results['tests']['streaming']['pass']
    else:
        all_passed = base_passed

    results['verdict'] = 'PASSED' if all_passed else ('WARNING' if base_passed else 'FAILED')

    print("\n" + "=" * 70 + f"\n结果: {results['verdict']}\n" + "=" * 70)

    # 保存结果
    result_file = os.path.join(output_dir, "results.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"详细报告: {result_file}")

    return results

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 命令行入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == '__main__':
    import argparse
    import subprocess

    parser = argparse.ArgumentParser(
        description="LightTTS CosyVoice2 部署验证",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python verify_deployment.py --full\n  python verify_deployment.py --server http://localhost:8081"
    )
    parser.add_argument('--server', '-s', default=DEFAULT_SERVER, help='服务器地址')
    parser.add_argument('--output', '-o', default='/tmp/lighttts_verify', help='输出目录')
    parser.add_argument('--full', '-f', action='store_true', help='完整测试')
    args = parser.parse_args()

    # 等待服务就绪
    print(f"\n等待服务就绪: {args.server}")
    for i in range(60):
        ok, _ = check_health(args.server)
        if ok:
            print("服务已就绪！\n")
            break
        print(f"\r  等待中... ({i+1}s)", end='')
        time.sleep(1)
    else:
        print("\n错误: 服务启动超时或不可达")
        sys.exit(1)

    # 运行测试
    result = run_tests(args.server, args.output, args.full)
    sys.exit(0 if result['verdict'] == 'PASSED' else 1)
