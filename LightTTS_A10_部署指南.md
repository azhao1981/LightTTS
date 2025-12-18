# LightTTS A10 显卡优化部署指南

## 概述

本指南针对 NVIDIA A10 (24GB VRAM) 显卡优化 LightTTS + CosyVoice2 的部署，实现首响时间(TTFA)和整体吞吐量(Throughput)的最佳平衡。

## 目录

1. [环境准备](#环境准备)
2. [代码修改](#代码修改)
3. [启动脚本](#启动脚本)
4. [性能优化建议](#性能优化建议)
5. [测试验证](#测试验证)

## 环境准备

### 1. 激活虚拟环境

```bash
cd ~/tts/
source .envrc
```

### 2. 设置环境变量

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)/cosyvoice"
```

## 代码修改

### 1. 修改 `light_tts/server/api_cli.py`

在文件末尾（第98行后）添加以下参数：

```python
parser.add_argument(
    "--data_type",
    type=str,
    default="float16",
    choices=["float16", "bfloat16", "float32"],
    help="Model precision. bfloat16 recommended for A10 Ampere architecture.",
)
```

### 2. 修改 `light_tts/server/tts_llm/manager.py`

在第157行的 `kvargs` 字典中添加：

```python
"data_type": getattr(self.args, 'data_type', "float16"),  # Pass data_type to model
```

### 3. 修改 `wsgi_wrapper.py` 的 CONFIG 部分

将 CONFIG 字典更新为以下 A10 优化配置：

```python
CONFIG = {
    "model_dir": "./pretrained_models/CosyVoice2-0.5B",
    "load_trt": True,
    "max_total_token_num": 32768,  # A10 optimized: reduced for 24GB
    "max_req_total_len": 8192,
    "host": "0.0.0.0",
    "port": 8080,
    "httpserver_workers": 2,  # A10 optimized: 2 workers for 24GB
    # Subprocess configuration
    "encode_process_num": 1,
    "decode_process_num": 2,  # A10 optimized: 2 decode processes
    "encode_paral_num": 32,  # A10 optimized: reduced to prevent OOM
    "gpt_paral_num": 32,
    "decode_paral_num": 2,  # Allow 2 concurrent decodes
    "decode_max_batch_size": 2,  # A10 optimized: batch size 2
    # Other defaults
    "zmq_mode": "ipc:///tmp/",
    "tokenizer_mode": "slow",
    "load_way": "HF",
    "running_max_req_size": 100,
    "trust_remote_code": False,
    "disable_log_stats": False,
    "log_stats_interval": 10,
    "router_token_ratio": 0.0,
    "router_max_new_token_len": 1024,
    "router_max_wait_tokens": 8,
    "cache_capacity": 200,
    "cache_reserved_ratio": 0.5,
    "sample_close": False,
    "health_monitor": False,
    "disable_cudagraph": False,
    "graph_max_batch_size": 16,
    "graph_max_len_in_batch": 32768,
    "load_jit": False,
    "batch_max_tokens": 4096,  # Control prefill batch size
    "graph_max_batch_size": 4,  # CUDA Graph optimization
    "mode": ["triton_flashdecoding"],  # Flash Attention for A10
    "data_type": "bfloat16",  # A10 optimized precision
}
```

在 `init_subprocesses` 函数中，第132行后添加：

```python
# Set data_type for models if specified
if hasattr(args, 'data_type') and args.data_type:
    os.environ["LIGHTTTS_DATA_TYPE"] = args.data_type
```

## 启动脚本

### 1. 创建 `run_a10_optimized.py`

```python
#!/usr/bin/env python3
"""A10 Optimized LightTTS Launcher"""

import os
import sys
import argparse

# Add cosyvoice to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cosyvoice'))

def main():
    parser = argparse.ArgumentParser(description='LightTTS A10 Optimized Launcher')
    parser.add_argument('--workers', type=int, default=2, help='Number of HTTP workers')
    parser.add_argument('--decode-proc', type=int, default=2, help='Number of decode processes')
    parser.add_argument('--precision', type=str, default='bfloat16',
                        choices=['float16', 'bfloat16', 'float32'],
                        help='Model precision (bfloat16 recommended for A10)')
    parser.add_argument('--model-dir', type=str,
                        default='./pretrained_models/CosyVoice2-0.5B',
                        help='Model directory')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=8080, help='Port number')

    args = parser.parse_args()

    # Print configuration
    print("=" * 60)
    print("LightTTS A10 Optimized Configuration")
    print("=" * 60)
    print(f"HTTP Workers: {args.workers}")
    print(f"Decode Processes: {args.decode_proc}")
    print(f"Precision: {args.precision}")
    print(f"Model: {args.model_dir}")
    print(f"Address: {args.host}:{args.port}")
    print("=" * 60)

    # Prepare sys.argv for the server
    sys.argv = [
        'api_server.py',
        '--host', args.host,
        '--port', str(args.port),
        '--model_dir', args.model_dir,
        '--httpserver_workers', str(args.workers),
        '--encode_process_num', '1',
        '--decode_process_num', str(args.decode_proc),
        '--encode_paral_num', '32',
        '--gpt_paral_num', '32',
        '--decode_paral_num', '2',
        '--decode_max_batch_size', '2',
        '--max_total_token_num', '32768',
        '--max_req_total_len', '8192',
        '--batch_max_tokens', '4096',
        '--graph_max_batch_size', '4',
        '--data_type', args.precision,
        '--load_trt', 'True',
        '--disable_cudagraph', 'False',
        '--mode', 'triton_flashdecoding',
        '--zmq_mode', 'ipc:///tmp/',
        '--tokenizer_mode', 'slow',
        '--load_way', 'HF',
        '--running_max_req_size', '100',
        '--trust_remote_code', 'False',
        '--disable_log_stats', 'False',
        '--log_stats_interval', '10',
        '--router_token_ratio', '0.0',
        '--router_max_new_token_len', '1024',
        '--router_max_wait_tokens', '8',
        '--cache_capacity', '200',
        '--cache_reserved_ratio', '0.5',
        '--sample_close', 'False',
        '--health_monitor', 'False',
        '--gpt_paral_step_num', '200',
        '--load_jit', 'False',
    ]

    # Import and run the server
    from light_tts.server.api_server import main
    main()

if __name__ == "__main__":
    main()
```

### 2. 创建 `start_a10.sh` (Bash脚本)

```bash
#!/bin/bash

# LightTTS A10 Optimized Startup Script
# Load virtual environment first
cd ~/tts/
source .envrc

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)/cosyvoice"

# Go to light-tts directory
cd light-tts

echo "========================================"
echo "Starting LightTTS with A10 optimizations"
echo "========================================"

# Run with optimized configuration
python run_a10_optimized.py "$@"
```

### 3. 设置执行权限

```bash
chmod +x run_a10_optimized.py
chmod +x start_a10.sh
```

## 启动命令

### 方式1：使用脚本启动（推荐）

```bash
# 默认配置（2 workers, BF16）
./start_a10.sh

# 最低延迟配置（1 worker, FP16）
./start_a10.sh --workers 1 --decode-proc 1 --precision float16

# 高吞吐配置（3 workers, 但可能OOM）
./start_a10.sh --workers 3 --decode-proc 3
```

### 方式2：直接使用api_server.py

```bash
cd ~/tts/
source .envrc
cd light-tts

python -m light_tts.server.api_server \
  --model_dir ./pretrained_models/CosyVoice2-0.5B \
  --host 0.0.0.0 \
  --port 8080 \
  --httpserver_workers 2 \
  --encode_process_num 1 \
  --decode_process_num 2 \
  --decode_max_batch_size 2 \
  --max_total_token_num 32768 \
  --data_type bfloat16 \
  --load_trt True \
  --mode triton_flashdecoding
```

## 性能优化建议

### 1. 精度选择
- **BF16 (推荐)**: A10原生支持，比FP16数值稳定性更好
- **FP16**: 如果内存紧张或需要极低延迟
- **FP32**: 不推荐，显存占用过大

### 2. 并发配置
- **低延迟模式**: 1个HTTP worker, 1个decode process
- **平衡模式**: 2个HTTP workers, 2个decode processes（默认）
- **高吞吐模式**: 3个HTTP workers, 3个decode processes（谨慎使用）

### 3. 内存优化参数
- `max_total_token_num`: 32768 (适合24GB显存)
- `batch_max_tokens`: 4096 (控制prefill batch)
- `decode_max_batch_size`: 2 (允许小批量处理)

### 4. 其他优化
- 始终启用 `--load_trt True` (TensorRT加速)
- 使用 `triton_flashdecoding` 模式 (Flash Attention)
- `graph_max_batch_size: 4` (CUDA Graph优化)

## 测试验证

### 1. 健康检查

```bash
curl http://localhost:8080/healthz
```

### 2. 简单TTS测试

需要准备一个参考音频文件 `test/prompt.wav`：

```bash
curl -X POST http://localhost:8080/inference_zero_shot \
  -F "prompt_audio=@test/prompt.wav" \
  -F "text=你好，这是A10优化测试" \
  -F "language=zh" \
  -F "streaming=true" \
  -o output.wav
```

### 3. 流式测试

```python
import requests
import aiohttp
import asyncio

async def test_streaming():
    url = "http://localhost:8080/inference_zero_shot"

    # Read prompt audio
    with open("test/prompt.wav", "rb") as f:
        prompt_audio = f.read()

    # Prepare request
    data = aiohttp.FormData()
    data.add_field('prompt_audio', prompt_audio,
                  filename='prompt.wav',
                  content_type='audio/wav')
    data.add_field('text', '你好，这是流式测试')
    data.add_field('streaming', 'true')
    data.add_field('language', 'zh')

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data) as response:
            if response.status == 200:
                first_chunk = True
                async for chunk in response.content.iter_chunked(4096):
                    if first_chunk:
                        print(f"First chunk received!")
                        first_chunk = False
                    # Process audio chunk here

# Run test
asyncio.run(test_streaming())
```

## 性能基准

### 预期性能指标（A10 24GB）

| 配置 | TTFA | 并发数 | 显存占用 | 吞吐量 |
|------|------|--------|----------|--------|
| 低延迟(1 worker) | 150-200ms | 1 | ~8GB | 低 |
| 平衡(2 workers) | 200-300ms | 5-10 | ~16GB | 中等 |
| 高吞吐(3 workers) | 300-400ms | 10+ | ~22GB | 高 |

### 监控命令

```bash
# 监控GPU使用情况
nvidia-smi -l 1

# 监控进程
ps aux | grep python

# 查看端口占用
netstat -tlnp | grep 8080
```

## 故障排除

### 1. 显存不足 (OOM)
- 减少 `--workers` 和 `--decode-proc` 数量
- 使用 `--precision float16`
- 降低 `max_total_token_num` 到 16384

### 2. 启动失败
- 检查虚拟环境是否正确加载
- 确认模型文件完整性
- 查看日志输出中的错误信息

### 3. 性能不佳
- 确认 TensorRT 已启用
- 检查是否使用了正确的精度模式
- 调整 batch size 和并行参数

## 补充说明

1. **关于 `.envrc`**
   - 该文件由 direnv 管理，用于自动加载虚拟环境
   - 如果没有安装 direnv，可以手动激活：
   ```bash
   conda activate tts-research  # 或您的环境名
   ```

2. **性能调优是一个迭代过程**
   - 从默认配置开始
   - 根据实际负载调整参数
   - 使用监控工具验证优化效果

3. **生产环境部署**
   - 考虑使用 Docker 容器化
   - 配置负载均衡器
   - 设置健康检查和自动重启

## 联系与支持

如遇到问题，请提供：
- 错误日志
- GPU 型号和驱动版本
- 使用的启动参数
- nvidia-smi 输出

---

*最后更新: 2024-12-18*