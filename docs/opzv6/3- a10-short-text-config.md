# NVIDIA A10 GPU 短文本处理配置推荐

## A10 GPU 规格

- **GPU Memory**: 24 GB GDDR6
- **CUDA Cores**: 10240
- **Tensor Cores**: 320 (Third-generation)
- **Memory Bandwidth**: 936 GB/s
- **TDP**: 150W

## 短文本处理特点

短文本处理（< 100 字）的特殊需求：
1. **极低延迟**：首字延迟 < 100ms
2. **高并发**：支持大量并发请求
3. **快速启动**：减少模型加载和初始化时间
4. **内存效率**：避免大批量处理导致的内存浪费

## 推荐配置

### 核心配置参数

```bash
#!/bin/bash
# scripts/a10-short-text.sh

# 模型路径
MODEL_DIR="./pretrained_models/CosyVoice2-0.5B-latest"

# A10 短文本优化配置
python -m light_tts.server.api_server \
  --model_dir ${MODEL_DIR} \
  --host 0.0.0.0 \
  --port 8080 \
  \
  --encode_process_num 2 \
  --encode_paral_num 4 \
  \
  --decode_process_num 2 \
  --decode_max_batch_size 1 \
  --decode_paral_num 3 \
  \
  --gpt_paral_num 80 \
  --gpt_paral_step_num 50 \
  \
  --max_total_token_num 16384 \
  --max_req_total_len 2048 \
  --batch_max_tokens 1024 \
  \
  --router_max_wait_tokens 2 \
  --router_max_new_token_len 512 \
  \
  --cache_capacity 500 \
  --cache_reserved_ratio 0.7 \
  \
  --load_trt True \
  --disable_cudagraph false \
  --graph_max_batch_size 2 \
  --graph_max_len_in_batch 512 \
  \
  --mode triton_flashdecoding,triton_int8kv \
  \
  --httpserver_workers 6 \
  --timeout_keep_alive 2 \
  \
  --disable_log_stats false \
  --log_stats_interval 5
```

### 配置参数说明

| 参数 | 推荐值 | 原因 |
|------|--------|------|
| `decode_process_num` | 2 | A10 可轻松处理 2 个 Decode workers |
| `decode_max_batch_size` | 1 | 短文本不需要大批次，避免延迟增加 |
| `decode_paral_num` | 3 | 提高并发处理能力 |
| `gpt_paral_num` | 80 | 平衡吞吐和延迟 |
| `gpt_paral_step_num` | 50 | 小步数，快速响应 |
| `max_total_token_num` | 16384 | 减少内存占用，专注短文本 |
| `max_req_total_len` | 2048 | 短文本长度限制 |
| `batch_max_tokens` | 1024 | 小批次，快速处理 |
| `router_max_wait_tokens` | 2 | 极短等待时间 |
| `cache_capacity` | 500 | 大缓存提高命中率 |
| `httpserver_workers` | 6 | 高并发 HTTP 处理 |

## 性能优化细节

### 1. TensorRT 优化

```python
# 确保 TensorRT 引擎已预构建
# 在首次启动时执行：
python -m light_tts.utils.build_trt \
  --model_dir ${MODEL_DIR} \
  --max_batch_size 2 \
  --max_seq_length 512
```

### 2. CUDA Graph 优化

```python
# 启用 CUDA Graph 捕获小推理图
# 配置文件：config/cuda_graph.yaml
cuda_graph:
  enabled: true
  max_batch_size: 2
  max_seq_length: 512
  warmup_iterations: 10
```

### 3. 模型量化

```bash
# 使用 INT8 量化减少内存使用
--mode triton_flashdecoding,triton_int8kv,triton_int8weight

# 或者使用 FP8 量化（A10 支持）
--mode triton_flashdecoding,triton_fp8w8a8
```

### 4. 内存管理优化

```python
# config/memory_optimization.py
import torch

def optimize_for_a10():
    """A10 GPU 专用内存优化"""

    # 设置内存分配策略
    torch.cuda.set_per_process_memory_fraction(0.85)  # 使用 85% 显存

    # 启用内存映射
    torch.multiprocessing.set_sharing_strategy('file_system')

    # 预分配内存池
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_per_process_memory_fraction(0.85, device=i)
        # 预热 GPU
        torch.cuda.empty_cache()

    # 设置缓存分配器
    torch.cuda.memory.set_per_process_memory_fraction(0.85)
```

## 短文本专用优化

### 1. 预加载常用模板

```python
# light_tts/server/short_text_cache.py
import asyncio
from typing import Dict, List

class ShortTextCache:
    """短文本专用缓存"""

    def __init__(self):
        self.templates = {}
        self.max_template_length = 50

    async def preload_templates(self):
        """预加载常用模板"""
        templates = [
            "您好",
            "谢谢",
            "再见",
            "请稍等",
            "正在处理中",
            # ... 更多常用短语
        ]

        for text in templates:
            # 预先生成并缓存
            cached_result = await self.preprocess_text(text)
            self.templates[text] = cached_result

    async def get_cached(self, text: str):
        """获取缓存的预处理结果"""
        return self.templates.get(text)
```

### 2. FastAPI 优化

```python
# light_tts/server/api_fast.py
from fastapi import FastAPI, BackgroundTasks
from fastapi.concurrency import run_in_threadpool

app = FastAPI()

# 短文本专用端点
@app.post("/inference_short")
async def inference_short(request: ShortTextRequest):
    """短文本快速推理端点"""

    # 快速路径：检查缓存
    cached_result = await short_text_cache.get_cached(request.text)
    if cached_result:
        return cached_result

    # 异步处理
    result = await run_in_threadpool(
        process_short_text,
        request.text,
        request.voice_template
    )

    return result
```

### 3. 连接池优化

```python
# light_tts/server/connection_pool.py
import aiohttp
from aiohttp import ClientSession

class OptimizedConnectionPool:
    """优化的连接池"""

    def __init__(self):
        self.session = None

    async def init(self):
        """初始化连接池"""
        connector = aiohttp.TCPConnector(
            limit=1000,  # 大量并发连接
            limit_per_host=100,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )

        timeout = aiohttp.ClientTimeout(
            total=5,  # 短超时
            connect=1,
            sock_connect=1,
            sock_read=2
        )

        self.session = ClientSession(
            connector=connector,
            timeout=timeout
        )
```

## 性能监控

### 关键指标

```python
# light_tts/server/short_text_metrics.py
from prometheus_client import Histogram, Gauge, Counter

# 短文本专用指标
SHORT_TEXT_LATENCY = Histogram(
    'short_text_latency_seconds',
    'Short text processing latency',
    buckets=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
)

FIRST_TOKEN_LATENCY = Histogram(
    'first_token_latency_seconds',
    'Time to first audio chunk',
    buckets=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
)

CONCURRENT_REQUESTS = Gauge(
    'short_text_concurrent_requests',
    'Number of concurrent short text requests'
)

CACHE_HIT_RATE = Gauge(
    'short_text_cache_hit_rate',
    'Short text cache hit rate'
)
```

### 实时监控脚本

```bash
#!/bin/bash
# scripts/monitor_a10.sh

while true; do
    clear
    echo "=== A10 GPU 短文本处理监控 ==="
    echo "时间: $(date)"
    echo

    # GPU 使用情况
    echo "GPU 状态:"
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
    while read util_mem util_gpu mem_used mem_total temp; do
        echo "  GPU: 利用率 ${util_gpu}% | 内存 ${util_mem}% | ${mem_used}MB/${mem_total}MB | 温度 ${temp}°C"
    done
    echo

    # 进程状态
    echo "进程状态:"
    ps aux | grep light_tts | grep -v grep | wc -l | xargs echo "  活跃进程数:"
    echo

    # TPS 和延迟
    echo "性能指标:"
    curl -s http://localhost:8080/metrics | grep -E "short_text_latency|first_token_latency|concurrent_requests" | \
    while read line; do
        echo "  $line"
    done
    echo

    sleep 2
done
```

## 压力测试

### 短文本压测脚本

```python
# test/short_text_stress.py
import asyncio
import aiohttp
import time
import numpy as np

async def short_text_stress_test():
    """短文本压力测试"""

    # 测试数据（10-50字的短文本）
    test_texts = [
        "你好",
        "欢迎使用语音合成服务",
        "今天天气真不错",
        "请稍等，正在为您处理",
        "您的请求已收到，请耐心等待",
        "这是一个测试消息",
        "语音合成技术越来越成熟了",
        "感谢您使用我们的服务"
    ]

    url = "http://localhost:8080/inference_short"

    async with aiohttp.ClientSession() as session:
        latencies = []
        first_token_times = []

        # 并发测试
        concurrent_users = 50
        requests_per_user = 100

        async def user_simulation(user_id):
            """模拟单个用户"""
            user_latencies = []

            for i in range(requests_per_user):
                text = test_texts[i % len(test_texts)]

                start_time = time.time()

                async with session.post(url, json={"text": text}) as resp:
                    if resp.status == 200:
                        # 记录首字节时间
                        first_byte_time = time.time()

                        # 读取响应
                        await resp.read()

                        total_time = (time.time() - start_time) * 1000
                        first_token = (first_byte_time - start_time) * 1000

                        user_latencies.append({
                            'total': total_time,
                            'first_token': first_token
                        })

            return user_latencies

        # 启动并发用户
        tasks = [user_simulation(i) for i in range(concurrent_users)]
        results = await asyncio.gather(*tasks)

        # 汇总结果
        for user_results in results:
            latencies.extend([r['total'] for r in user_results])
            first_token_times.extend([r['first_token'] for r in user_results])

        # 统计分析
        print("\n=== 短文本压力测试结果 ===")
        print(f"总请求数: {len(latencies)}")
        print(f"并发用户: {concurrent_users}")
        print(f"每用户请求数: {requests_per_user}")
        print()

        # 延迟统计
        print("端到端延迟 (ms):")
        print(f"  平均: {np.mean(latencies):.2f}")
        print(f"  P50: {np.percentile(latencies, 50):.2f}")
        print(f"  P90: {np.percentile(latencies, 90):.2f}")
        print(f"  P95: {np.percentile(latencies, 95):.2f}")
        print(f"  P99: {np.percentile(latencies, 99):.2f}")
        print()

        print("首字延迟 (ms):")
        print(f"  平均: {np.mean(first_token_times):.2f}")
        print(f"  P50: {np.percentile(first_token_times, 50):.2f}")
        print(f"  P90: {np.percentile(first_token_times, 90):.2f}")
        print(f"  P95: {np.percentile(first_token_times, 95):.2f}")
        print(f"  P99: {np.percentile(first_token_times, 99):.2f}")
        print()

        # QPS
        total_time = max(latencies) / 1000  # 秒
        qps = len(latencies) / total_time
        print(f"QPS: {qps:.2f}")

if __name__ == "__main__":
    asyncio.run(short_text_stress_test())
```

## 性能预期

基于 A10 GPU 的短文本处理性能预期：

| 指标 | 目标值 | 优秀值 |
|------|--------|--------|
| 首字延迟 | < 100ms | < 50ms |
| 端到端延迟 | < 200ms | < 100ms |
| QPS | > 200 | > 500 |
| 并发用户 | 50+ | 100+ |
| GPU 利用率 | > 60% | > 80% |
| 内存使用 | < 16GB | < 12GB |

## 故障排查

### 常见问题

1. **延迟突然增加**
   ```bash
   # 检查 GPU 温度
   nvidia-smi --query-gpu=temperature.gpu --format=csv

   # 检查是否有长文本占用资源
   curl http://localhost:8080/metrics | grep request_length
   ```

2. **内存泄漏**
   ```python
   # 监控内存使用
   import torch
   import psutil

   def monitor_memory():
       while True:
           gpu_mem = torch.cuda.memory_allocated() / 1024**3
           cpu_mem = psutil.virtual_memory().percent
           print(f"GPU: {gpu_mem:.2f}GB, CPU: {cpu_mem}%")
           time.sleep(5)
   ```

3. **性能下降**
   ```bash
   # 重置 CUDA 缓存
   python -c "import torch; torch.cuda.empty_cache()"

   # 检查 TensorRT 引擎
   ls -la ./model_cache/trt_engines/
   ```

## 总结

A10 GPU 非常适合短文本处理场景。通过以上优化配置，可以实现：
- 超低延迟的首字响应（< 50ms）
- 高并发处理能力（500+ QPS）
- 优秀的资源利用率（GPU > 80%）
- 稳定的长时间运行

建议定期监控性能指标，根据实际负载调整参数，保持最佳性能。