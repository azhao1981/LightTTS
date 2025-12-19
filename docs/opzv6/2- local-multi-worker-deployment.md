# Light-TTS 本地多 Worker 部署指南 (修复版)

## 概述

本文档详细介绍如何在单台服务器上部署 1个 LLM + 2-3个 Flow+HiFi-GAN Worker 的配置。**注意：需要先修复 decode_process_num 限制问题**（参考 `decode-process-limit-analysis.md`）。

## 🚨 部署前必读

### 修复限制问题

在部署多worker之前，必须先修复代码中的限制：

1. **修改 api_start.py**：
   ```python
   # 注释或删除第111-112行的限制
   # num_loras = 1
   # assert args.decode_process_num <= num_loras

   # 修改第142-143行的端口分配
   # 原代码：
   # tts_decode_ports = can_use_ports[0 : num_loras]
   # 修改为：
   tts_decode_ports = can_use_ports[0 : args.decode_process_num]
   ```

2. **验证修复**：
   ```bash
   # 应该可以成功启动多个workers
   python -m light_tts.server.api_server --decode_process_num 3 --model_dir /path/to/model
   ```

## 架构概览

### 本地多进程架构（修复后）

```
┌─────────────────────────────────────────────────────────┐
│                    HTTP Server                          │
│              (FastAPI + Gunicorn)                      │
│                    Workers: 3                          │
└─────────────┬───────────────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────────┐
│              HttpServerManager                          │
│                   (进程间协调)                           │
└─────┬───────────┬───────────────────┬─────────────────────┘
      │           │                   │
┌─────▼───┐ ┌─────▼─────┐   ┌───────▼───────┐ ┌──────────┐
│ Encode  │ │    LLM    │   │  Decode 1     │ │ Decode 2 │
│Process 1│ │ Process 1 │   │ Process 1     │ │ Process 2 │
│         │ │ (GPU 0)   │   │ (GPU 1)       │ │ (GPU 2)   │
└─────────┘ └───────────┘   └───────────────┘ └──────────┘
```

### 数据流设计

```
HTTP Request → HttpServer → Encode → LLM → (负载均衡) → Decode Workers
                                          ↓
                                    ┌─────────────┐
                                    │ Decode 1    │ ← GPU 1
                                    │ (Port 8084) │
                                    └─────────────┘
                                          ↓
                                    ┌─────────────┐
                                    │ Decode 2    │ ← GPU 2
                                    │ (Port 8085) │
                                    └─────────────┘
```

### 进程职责

1. **HTTP Server**：处理外部请求，使用 Gunicorn 多进程提高并发
2. **Encode Process**：文本预处理和说话人特征提取（CPU密集）
3. **LLM Process**：语音 token 生成（GPU 密集）
4. **Decode Processes**：Flow+HiFi-GAN 音频合成（GPU 密集，多进程）

## 部署配置

### 基础配置（2个 Decode Workers）

```bash
#!/bin/bash
# scripts/start-2workers.sh

# 环境变量
export CUDA_VISIBLE_DEVICES=0,1,2
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 模型路径
MODEL_DIR="./pretrained_models/CosyVoice2-0.5B-latest"

# 启动命令
python -m light_tts.server.api_server \
  --model_dir ${MODEL_DIR} \
  --host 0.0.0.0 \
  --port 8080 \
  --encode_process_num 1 \
  --encode_paral_num 4 \
  --decode_process_num 2 \
  --decode_paral_num 2 \
  --decode_max_batch_size 4 \
  --gpt_paral_num 100 \
  --gpt_paral_step_num 200 \
  --max_total_token_num 65536 \
  --max_req_total_len 32768 \
  --batch_max_tokens 8192 \
  --router_max_wait_tokens 8 \
  --cache_capacity 200 \
  --load_trt True \
  --mode triton_flashdecoding \
  --httpserver_workers 3
```

### 高性能配置（3个 Decode Workers）

```bash
#!/bin/bash
# scripts/start-3workers.sh

# 环境变量
export CUDA_VISIBLE_DEVICES=0,1,2
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 模型路径
MODEL_DIR="./pretrained_models/CosyVoice2-0.5B-latest"

# 启动命令
python -m light_tts.server.api_server \
  --model_dir ${MODEL_DIR} \
  --host 0.0.0.0 \
  --port 8080 \
  --encode_process_num 1 \
  --encode_paral_num 6 \
  --decode_process_num 3 \
  --decode_paral_num 2 \
  --decode_max_batch_size 3 \
  --gpt_paral_num 150 \
  --gpt_paral_step_num 300 \
  --max_total_token_num 98304 \
  --max_req_total_len 32768 \
  --batch_max_tokens 12288 \
  --router_max_wait_tokens 6 \
  --cache_capacity 300 \
  --load_trt True \
  --mode triton_flashdecoding,triton_int8kv \
  --httpserver_workers 4 \
  --log_stats_interval 10
```

### Gunicorn 生产配置

```python
# wsgi_wrapper.py
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 修改启动参数
if len(sys.argv) > 1:
    pass
else:
    # 默认生产配置
    sys.argv.extend([
        "--model_dir", "./pretrained_models/CosyVoice2-0.5B-latest",
        "--decode_process_num", "3",
        "--decode_max_batch_size", "3",
        "--gpt_paral_num", "150",
        "--load_trt", "True",
        "--httpserver_workers", "4"
    ])

# 启动应用
from light_tts.server.api_http import app
```

```bash
#!/bin/bash
# run.sh

# 性能优化环境变量
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN

# Gunicorn 配置
WORKERS=4                    # HTTP worker 进程数
WORKER_CLASS=uvicorn.workers.UvicornWorker
HOST=0.0.0.0
PORT=8080
TIMEOUT=300
KEEPALIVE=5

# 启动命令
exec gunicorn -w $WORKERS \
    -k $WORKER_CLASS \
    -b $HOST:$PORT \
    --timeout $TIMEOUT \
    --keep-alive $KEEPALIVE \
    --access-logfile - \
    --error-logfile - \
    --preload \
    wsgi_wrapper:app
```

## GPU 配置优化

### GPU 资源分配

```python
# 自动 GPU 分配逻辑（已在代码中实现）
def get_gpu_id(decode_proc_index, total_gpus):
    """为 Decode 进程分配 GPU"""
    return decode_proc_index % total_gpus

# 示例分配方案
# 3 个 GPU, 3 个 Decode Workers:
# - Decode 0 -> GPU 0 (LLM 也在 GPU 0)
# - Decode 1 -> GPU 1
# - Decode 2 -> GPU 2
```

### 内存优化

```python
# config/memory_config.py
import torch

def optimize_gpu_memory():
    """优化 GPU 内存使用"""
    # 启用内存映射
    torch.multiprocessing.set_sharing_strategy('file_system')

    # 设置内存分配策略
    torch.cuda.set_per_process_memory_fraction(0.9)  # 使用90% GPU内存

    # 启用缓存分配器
    torch.cuda.empty_cache()

    # 设置内存池
    torch.cuda.memory.set_per_process_memory_fraction(0.95)

# GPU 内存监控
def monitor_gpu_memory():
    """监控 GPU 内存使用"""
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3

        print(f"GPU {i}: {allocated:.2f}GB / {total:.2f}GB ({allocated/total*100:.1f}%)")
```

## 性能调优

### 参数说明与调优建议

| 参数 | 默认值 | 建议值 | 说明 |
|------|--------|--------|------|
| `decode_process_num` | 1 | 2-3 | Decode 进程数，根据 GPU 数量设置 |
| `decode_max_batch_size` | 1 | 2-4 | 批大小，影响吞吐和延迟 |
| `decode_paral_num` | 1 | 2-4 | 每个 Decode 进程的并行度 |
| `gpt_paral_num` | 50 | 100-200 | LLM 推理并行度 |
| `batch_max_tokens` | None | 8192-16384 | 批处理最大 token 数 |
| `router_max_wait_tokens` | 8 | 4-8 | 调度等待时间，越小延迟越低 |

### 场景优化配置

#### 高吞吐配置（长文本）

```bash
# 优化批处理，提高吞吐量
--decode_process_num 3
--decode_max_batch_size 4
--gpt_paral_num 200
--batch_max_tokens 16384
--router_max_wait_tokens 12
```

#### 低延迟配置（短文本/实时）

```bash
# 减少批处理，降低延迟
--decode_process_num 3
--decode_max_batch_size 2
--gpt_paral_num 100
--batch_max_tokens 4096
--router_max_wait_tokens 4
--gpt_paral_step_num 100
```

#### 内存受限配置

```bash
# 减少内存使用
--decode_process_num 2
--decode_max_batch_size 2
--max_total_token_num 32768
--mode ppl_int8kv  # 启用 KV Cache 量化
```

## 监控与诊断

### 性能监控脚本

```python
# scripts/monitor.py
import psutil
import torch
import time
import requests

def monitor_system():
    """监控系统资源"""
    while True:
        # CPU 使用率
        cpu_percent = psutil.cpu_percent(interval=1)

        # 内存使用
        memory = psutil.virtual_memory()

        # GPU 使用率
        gpu_stats = []
        for i in range(torch.cuda.device_count()):
            gpu_stats.append({
                'id': i,
                'memory_used': torch.cuda.memory_allocated(i) / 1024**3,
                'memory_total': torch.cuda.get_device_properties(i).total_memory / 1024**3,
                'utilization': get_gpu_utilization(i)
            })

        # TPS (每秒处理请求数)
        tps = get_current_tps()

        # 打印监控信息
        print(f"\n{'='*60}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"CPU: {cpu_percent:.1f}%")
        print(f"Memory: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB)")

        for gpu in gpu_stats:
            print(f"GPU {gpu['id']}: {gpu['utilization']:.1f}% | "
                  f"Memory: {gpu['memory_used']:.1f}/{gpu['memory_total']:.1f}GB")

        print(f"TPS: {tps:.1f}")

        time.sleep(5)

def get_gpu_utilization(gpu_id):
    """获取 GPU 利用率（需要 nvidia-ml-py）"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return util.gpu
    except:
        return 0

def get_current_tps():
    """获取当前 TPS"""
    try:
        # 从 Prometheus 获取指标
        response = requests.get('http://localhost:8080/metrics')
        # 解析指标返回 TPS
        return 0
    except:
        return 0

if __name__ == "__main__":
    monitor_system()
```

### 日志分析

```bash
# 提取性能日志
grep "cost_time" logs/light-tts.log | \
  awk '{print $NF}' | \
  sed 's/ms//' | \
  sort -n | \
  awk 'BEGIN{count=0; sum=0} {count++; sum+=$1; arr[NR]=$1}
  END{print "Count:", count;
       print "Avg:", sum/count "ms";
       print "P50:", arr[int(NR*0.5)] "ms";
       print "P90:", arr[int(NR*0.9)] "ms";
       print "P99:", arr[int(NR*0.99)] "ms"}'
```

## 故障处理

### 常见问题与解决方案

#### 1. GPU 内存不足

```bash
# 错误：CUDA out of memory
# 解决方案：
# 1. 减少批大小
--decode_max_batch_size 2

# 2. 启用量化
--mode ppl_int8kv

# 3. 减少 token 容量
--max_total_token_num 32768
```

#### 2. 进程卡死

```bash
# 检查进程状态
ps aux | grep light_tts

# 检查 GPU 进程
nvidia-smi pmon -s u

# 重启服务
systemctl restart light-tts
```

#### 3. 性能下降

```bash
# 检查 GPU 利用率
nvidia-smi dmon -s pucvmet

# 检查队列堆积
curl http://localhost:8080/metrics | grep queue

# 清理 GPU 缓存
python -c "import torch; torch.cuda.empty_cache()"
```

## 自动化部署脚本

### 一键部署脚本

```bash
#!/bin/bash
# scripts/deploy.sh

set -e

# 配置参数
MODEL_DIR=${1:-"./pretrained_models/CosyVoice2-0.5B-latest"}
DECODE_WORKERS=${2:-3}
HTTP_WORKERS=${3:-4}

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# 检查依赖
check_dependencies() {
    log "检查依赖..."

    # 检查 Python
    if ! command -v python &> /dev/null; then
        error "Python 未安装"
    fi

    # 检查 CUDA
    if ! command -v nvidia-smi &> /dev/null; then
        error "NVIDIA driver 未安装"
    fi

    # 检查 GPU 数量
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [ $GPU_COUNT -lt $((DECODE_WORKERS)) ]; then
        warn "GPU 数量 ($GPU_COUNT) 少于 Decode Workers ($DECODE_WORKERS)"
    fi

    # 检查模型文件
    if [ ! -d "$MODEL_DIR" ]; then
        error "模型目录不存在: $MODEL_DIR"
    fi
}

# 创建配置文件
create_config() {
    log "创建配置文件..."

    cat > config/production.yaml << EOF
model_dir: "$MODEL_DIR"
decode_process_num: $DECODE_WORKERS
httpserver_workers: $HTTP_WORKERS
decode_max_batch_size: 3
gpt_paral_num: 150
load_trt: true
mode: ["triton_flashdecoding"]
log_level: "INFO"
EOF
}

# 启动服务
start_service() {
    log "启动 Light-TTS 服务..."

    # 设置环境变量
    export CUDA_VISIBLE_DEVICES=0,1,2
    export OMP_NUM_THREADS=8

    # 启动服务
    ./run.sh > logs/light-tts.log 2>&1 &

    # 保存 PID
    echo $! > /var/run/light-tts.pid

    # 等待服务启动
    sleep 10

    # 健康检查
    if curl -f http://localhost:8080/healthz > /dev/null 2>&1; then
        log "服务启动成功！"
    else
        error "服务启动失败"
    fi
}

# 主流程
main() {
    log "开始部署 Light-TTS..."

    check_dependencies
    create_config
    start_service

    log "部署完成！"
    log "服务地址: http://localhost:8080"
    log "健康检查: http://localhost:8080/healthz"
    log "监控指标: http://localhost:8080/metrics"
}

main "$@"
```

### systemd 服务配置

```ini
# /etc/systemd/system/light-tts.service
[Unit]
Description=Light-TTS Service
After=network.target

[Service]
Type=forking
User=tts
Group=tts
WorkingDirectory=/opt/light-tts
ExecStart=/opt/light-tts/scripts/deploy.sh
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10

# 环境变量
Environment=CUDA_VISIBLE_DEVICES=0,1,2
Environment=PYTHONPATH=/opt/light-tts

# 资源限制
LimitNOFILE=65536
LimitNPROC=4096

# 日志
StandardOutput=append:/var/log/light-tts/access.log
StandardError=append:/var/log/light-tts/error.log

[Install]
WantedBy=multi-user.target
```

## 性能基准

### 测试脚本

```python
# test/benchmark.py
import asyncio
import aiohttp
import time
import statistics

async def benchmark():
    """性能基准测试"""
    url = "http://localhost:8080/inference_zero_shot"

    # 测试数据
    test_data = {
        "tts_text": "这是一个性能测试",
        "prompt_text": "请用自然的声音朗读",
        "prompt_wav": "base64_encoded_audio_here"
    }

    async with aiohttp.ClientSession() as session:
        latencies = []

        # 并发测试
        for i in range(100):
            start_time = time.time()

            async with session.post(url, json=test_data) as resp:
                if resp.status == 200:
                    audio_data = await resp.read()
                    latency = (time.time() - start_time) * 1000
                    latencies.append(latency)

                    if i % 10 == 0:
                        print(f"Request {i}: {latency:.2f}ms")
                else:
                    print(f"Error: {resp.status}")

        # 统计结果
        print("\n=== Performance Report ===")
        print(f"Total Requests: {len(latencies)}")
        print(f"Avg Latency: {statistics.mean(latencies):.2f}ms")
        print(f"P50 Latency: {statistics.median(latencies):.2f}ms")
        print(f"P90 Latency: {sorted(latencies)[int(len(latencies)*0.9)]:.2f}ms")
        print(f"P99 Latency: {sorted(latencies)[int(len(latencies)*0.99)]:.2f}ms")
        print(f"Min Latency: {min(latencies):.2f}ms")
        print(f"Max Latency: {max(latencies):.2f}ms")

if __name__ == "__main__":
    asyncio.run(benchmark())
```

## 总结

本地多 Worker 部署方案能够充分利用单机多 GPU 资源，通过合理的进程分配和参数调优，可以获得接近线性的性能提升。建议根据实际业务场景选择合适的配置参数，并持续监控和优化。