# 性能调优指南

## 概述

Light TTS 通过多级优化策略实现高性能语音合成，包括硬件加速、算法优化和系统调优。本指南详细介绍各种性能优化技术。

## 硬件配置建议

### GPU 配置

| GPU型号 | 显存 | 建议配置 | 性能表现 |
|---------|------|----------|----------|
| RTX 4090 | 24GB | 3 workers, decode_process_num=2 | 最佳性能 |
| RTX 3090 | 24GB | 3 workers, decode_process_num=2 | 高性能 |
| A100 | 40GB | 4 workers, decode_process_num=3 | 最高性能 |
| RTX 3080 | 10GB | 1 worker, decode_process_num=1 | 基础性能 |

### 内存配置

```bash
# 最小内存要求
系统内存: 16GB
GPU显存: 8GB

# 推荐内存配置
系统内存: 32GB+
GPU显存: 24GB+
```

## 参数调优

### 1. 并行度调优

#### Encode 并行配置
```python
CONFIG = {
    "encode_process_num": 2,        # Encode进程数
    "encode_paral_num": 4,          # 每个进程的并行任务数
    "encode_max_batch_size": 8,     # 最大批处理大小
}
```

**调优策略**:
- `encode_process_num`: 根据 CPU核心数设置，通常为 CPU核心数/2
- `encode_paral_num`: GPU内存充足时可增加，建议 4-8
- `encode_max_batch_size`: 影响内存使用，建议 4-16

#### LLM 并行配置
```python
CONFIG = {
    "gpt_paral_num": 100,           # LLM推理并行度
    "batch_max_tokens": 8192,       # 批处理最大token数
    "router_max_wait_tokens": 512,  # 最大等待token数
}
```

**调优策略**:
- `gpt_paral_num`: 控制并发推理数量，高吞吐场景可设置 200+
- `batch_max_tokens`: 影响内存和延迟，建议 4096-16384
- `router_max_wait_tokens`: 平衡延迟和吞吐量

#### Decode 并行配置
```python
CONFIG = {
    "decode_process_num": 2,        # Decode进程数
    "decode_paral_num": 2,          # 每个进程的并行任务数
    "decode_max_batch_size": 4,     # 最大批处理大小
    "graph_max_batch_size": 8,      # CUDA Graph最大批大小
}
```

**调优策略**:
- `decode_process_num`: 受num_loras限制，目前最大为1
- `decode_paral_num`: GPU内存允许时建议 2-4
- `decode_max_batch_size`: 关键参数，建议 2-8
- `graph_max_batch_size`: CUDA Graph优化，建议 4-16

### 2. 内存管理优化

#### Token容量配置
```python
CONFIG = {
    "max_total_token_num": 65536,   # 总token容量
    "max_req_total_len": 32768,     # 单请求最大长度
}
```

**内存计算公式**:
```
GPU内存需求 ≈ max_total_token_num * 2 bytes (FP16) * 模型参数量 + 激活内存
```

#### 缓存配置
```python
CONFIG = {
    "cache_capacity": 100,          # LRU缓存容量
    "kv_cache_mode": "int8",        # KV Cache量化模式
    "enable_kv_cache_quant": True,  # 启用KV Cache量化
}
```

### 3. 量化优化

#### 量化模式选择
```python
# INT8权重量化
quant_config = {
    "quant_type": "triton_int8weight"
}

# FP8精度
quant_config = {
    "quant_type": "triton-fp8w8a8-b128"
}

# 混合量化
quant_config = {
    "quant_type": "custom",
    "custom_cfg_path": "./quant_config.yaml"
}
```

#### 量化效果对比

| 量化模式 | 显存节省 | 性能影响 | 音质损失 | 推荐场景 |
|----------|----------|----------|----------|----------|
| FP32 | 0% | 基准 | 无 | 开发调试 |
| FP16 | 50% | 提升20% | 极小 | 生产推荐 |
| INT8 | 75% | 提升40% | 轻微 | 高并发 |
| FP8 | 80% | 提升50% | 轻微 | 最高性能 |

## 硬件加速

### 1. TensorRT 优化

#### 启用TensorRT
```bash
# 启动命令
python -m light_tts.server.api_server \
  --model_dir ./pretrained_models/CosyVoice2-0.5B-latest \
  --load_trt True \
  --trt_max_batch_size 8
```

#### TensorRT 配置
```python
CONFIG = {
    "load_trt": True,               # 启用TensorRT
    "trt_max_batch_size": 8,        # TRT最大批大小
    "trt_precision": "fp16",        # TRT精度模式
    "trt_workspace_size": 1073741824, # TRT工作空间(1GB)
}
```

### 2. CUDA Graph 优化

```python
CONFIG = {
    "disable_cudagraph": False,     # 启用CUDA Graph
    "graph_max_batch_size": 8,      # Graph优化批大小
    "graph_max_len_in_batch": 512,  # Graph优化序列长度
}
```

**优化效果**:
- 减少 GPU kernel启动开销
- 提升小批次推理性能 15-30%
- 特别适合流式处理场景

### 3. 自定义CUDA内核

#### Flash Attention优化
```python
CONFIG = {
    "use_flash_attention": True,    # 启用Flash Attention
    "fa_kernel_type": "triton",     # Triton内核
}
```

#### GQA优化
```python
CONFIG = {
    "enable_gqa": True,             # 启用GQA
    "gqa_group_size": 4,            # GQA分组大小
}
```

## 系统级优化

### 1. 操作系统调优

#### Linux内核参数
```bash
# 增加网络缓冲区
echo 'net.core.rmem_max = 134217728' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' >> /etc/sysctl.conf

# 增加文件描述符限制
echo '* soft nofile 65536' >> /etc/security/limits.conf
echo '* hard nofile 65536' >> /etc/security/limits.conf

# 优化内存管理
echo 'vm.swappiness = 10' >> /etc/sysctl.conf
```

#### GPU优化
```bash
# 设置GPU性能模式
nvidia-smi -pm 1

# 设置GPU时钟频率
nvidia-smi -ac 877,1215

# 禁用ECC（如果不需要）
nvidia-smi -e 0
```

### 2. Docker优化

#### 容器配置
```yaml
# docker-compose.yml
version: '3.8'
services:
  light-tts:
    deploy:
      resources:
        limits:
          memory: 32G
          nvidia.com/gpu: 1
        reservations:
          memory: 16G
          nvidia.com/gpu: 1
    shm_size: '4g'                 # 增加共享内存
    ulimits:
      memlock: -1
      stack: 67108864
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - OMP_NUM_THREADS=8
      - NCCL_IB_DISABLE=1
```

### 3. 进程调度优化

#### CPU亲和性
```python
import os
import multiprocessing as mp

# 设置CPU亲和性
def set_cpu_affinity(pid, cpu_list):
    os.sched_setaffinity(pid, cpu_list)

# 绑定不同进程到不同CPU核心
cpu_cores = list(range(mp.cpu_count()))
encode_cpus = cpu_cores[:len(cpu_cores)//2]
llm_cpus = cpu_cores[len(cpu_cores)//2:]

set_cpu_affinity(encode_pid, encode_cpus)
set_cpu_affinity(llm_pid, llm_cpus)
```

#### NUMA优化
```bash
# 查看NUMA拓扑
numactl --hardware

# 绑定进程到NUMA节点
numactl --cpunodebind=0 --membind=0 python app.py
```

## 网络优化

### 1. 传输优化

#### HTTP/2和Keep-Alive
```python
# Gunicorn配置
CONFIG = {
    "workers": 3,
    "worker_class": "uvicorn.workers.UvicornWorker",
    "worker_connections": 1000,
    "max_requests": 10000,
    "max_requests_jitter": 1000,
    "timeout": 300,
    "keepalive": 2,
}
```

#### WebSocket优化
```python
CONFIG = {
    "ws_max_size": 16777216,        # 16MB
    "ws_ping_interval": 20,         # 20秒ping间隔
    "ws_ping_timeout": 20,          # 20秒ping超时
    "ws_max_queue": 1024,           # 最大消息队列
}
```

### 2. 负载均衡

#### Nginx配置优化
```nginx
upstream tts_backend {
    least_conn;
    server 127.0.0.1:8080 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8081 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8082 max_fails=3 fail_timeout=30s;

    keepalive 32;
}

server {
    listen 80;

    # 优化缓冲区
    client_body_buffer_size 128k;
    client_max_body_size 50m;
    proxy_buffering on;
    proxy_buffer_size 4k;
    proxy_buffers 8 4k;

    # 启用压缩
    gzip on;
    gzip_types application/json;

    location / {
        proxy_pass http://tts_backend;
        proxy_set_header Connection "";
        proxy_http_version 1.1;

        # 超时设置
        proxy_connect_timeout 5s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}
```

## 性能监控

### 1. 关键指标

#### GPU监控
```bash
# 实时监控
nvidia-smi -l 1

# 详细指标
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv -l 1
```

#### 系统监控
```bash
# CPU和内存
htop
iostat -x 1

# 网络监控
iftop -i eth0
```

#### 应用监控
```python
# Prometheus指标
curl http://localhost:8080/metrics

# 关键指标
# lightllm_request_latency_seconds_bucket
# lightllm_request_count_total
# lightllm_gpu_memory_usage_bytes
```

### 2. 性能分析工具

#### PyTorch Profiler
```python
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
) as profiler:
    for batch in dataloader:
        # 推理代码
        output = model(batch)
        profiler.step()
```

#### NVIDIA Nsight
```bash
# 性能分析
nsys python -m light_tts.server.api_server --model_dir ./models

# 内存分析
nsys profile --stats=true python app.py
```

## 故障排除

### 1. 性能问题诊断

#### GPU利用率低
```bash
# 检查GPU状态
nvidia-smi

# 常见原因：
# 1. batch size太小
# 2. CPU瓶颈
# 3. 数据加载慢
```

#### 内存不足
```bash
# 检查内存使用
free -h
nvidia-smi

# 解决方案：
# 1. 减少batch size
# 2. 启用量化
# 3. 增加交换空间
```

### 2. 性能调优检查清单

#### 启动前检查
- [ ] TensorRT已编译 (`--load_trt True`)
- [ ] 量化配置正确
- [ ] GPU驱动版本匹配
- [ ] 系统参数已优化

#### 运行时监控
- [ ] GPU利用率 > 80%
- [ ] GPU显存使用合理
- [ ] 请求延迟在预期范围
- [ ] 错误率 < 1%

#### 定期优化
- [ ] 清理GPU缓存
- [ ] 更新驱动和CUDA
- [ ] 监控模型漂移
- [ ] 调整并发参数

## 基准测试

### 1. 性能基准

#### 单实例性能 (RTX 4090)
```
配置: decode_max_batch_size=4, gpt_paral_num=100
并发: 10个请求
平均延迟: 1.2秒
吞吐量: 8.3 requests/s
GPU利用率: 85%
显存使用: 7.2GB
```

#### 多实例扩展 (24GB GPU)
```
实例数: 3个
总并发: 30个请求
总吞吐量: 24 requests/s
平均延迟: 1.3秒
资源利用率: 90%
```

### 2. 压力测试

#### 测试脚本
```python
import asyncio
import aiohttp
import time

async def stress_test(base_url, num_requests=100, concurrency=10):
    semaphore = asyncio.Semaphore(concurrency)

    async def single_request(session, text):
        async with semaphore:
            start_time = time.time()
            # 执行TTS请求
            latency = time.time() - start_time
            return latency

    async with aiohttp.ClientSession() as session:
        tasks = [
            single_request(session, f"测试文本{i}")
            for i in range(num_requests)
        ]
        latencies = await asyncio.gather(*tasks)

        avg_latency = sum(latencies) / len(latencies)
        throughput = num_requests / max(latencies)

        print(f"平均延迟: {avg_latency:.2f}s")
        print(f"吞吐量: {throughput:.2f} requests/s")

# 运行测试
asyncio.run(stress_test("http://localhost:8080"))
```