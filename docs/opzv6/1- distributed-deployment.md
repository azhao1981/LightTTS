# Light-TTS 分布式分离部署指南

## 概述

本文档详细介绍如何将 Light-TTS 的 Flow Model + HiFi-GAN 作为 Frontend，LLM (Qwen2) 作为 Backend 进行分离部署。这种架构允许独立扩展各个组件，优化资源利用率。

## 架构设计

### 整体架构图

```
                    ┌─────────────────┐
                    │  Load Balancer  │
                    │   (Nginx/HAProxy)│
                    └─────────┬───────┘
                              │
                    ┌─────────▼───────┐
                    │  HTTP Gateway   │
                    │  (FastAPI)      │
                    └─────────┬───────┘
                              │
                    ┌─────────▼───────┐
                    │  Encode Module  │
                    │  (Frontend)     │
                    └─────────┬───────┘
                              │
                    ┌─────────▼───────┐    HTTP/gRPC/ZMQ
                    │   LLM Module    │◄─────────────────┐
                    │  (Qwen2 Backend)│                  │
                    └─────────┬───────┘                  │
                              │                          │
                    ┌─────────▼───────┐                  │
                    │ Flow + HiFi-GAN │                  │
                    │   (Frontend)    │                  │
                    └─────────────────┘                  │
                                                     ┌──▼──┐
                                                     │Redis│
                                                     │Cache│
                                                     └─────┘
```

### 服务拆分策略

1. **LLM Backend Service**
   - 负责文本到语音 token 的生成
   - GPU 密集型，需要高性能计算资源
   - 支持 Continuous Batching 和 KV Cache

2. **Flow+HiFi-GAN Frontend Service**
   - 负责 token 到音频的转换
   - 可与多个 LLM Backend 连接
   - 支持 TensorRT 加速

3. **Encode Service**
   - 文本预处理和说话人特征提取
   - CPU 密集型，可独立部署

4. **共享存储**
   - Redis：分布式缓存
   - 对象存储：模型文件和中间结果

## 实施步骤

### 第一阶段：网络抽象层改造

#### 1.1 创建传输抽象接口

```python
# light_tts/server/transport/transport_interface.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class DataTransport(ABC):
    """数据传输抽象接口"""

    @abstractmethod
    async def send(self, data: Any, target: Optional[str] = None) -> bool:
        """发送数据"""
        pass

    @abstractmethod
    async def recv(self, timeout: Optional[float] = None) -> Any:
        """接收数据"""
        pass

    @abstractmethod
    async def broadcast(self, data: Any) -> int:
        """广播数据到所有节点"""
        pass

# light_tts/server/transport/zmq_transport.py
import zmq.asyncio
import pickle

class ZMQTransport(DataTransport):
    def __init__(self, mode: str, address: str):
        self.context = zmq.asyncio.Context()
        self.mode = mode
        self.address = address

    async def send(self, data: Any, target: Optional[str] = None) -> bool:
        socket = self.context.socket(zmq.PUSH if self.mode == "push" else zmq.PUB)
        socket.connect(target or self.address)
        await socket.send_pyobj(data, protocol=pickle.HIGHEST_PROTOCOL)
        return True

# light_tts/server/transport/grpc_transport.py
import grpc
from generated import tts_pb2, tts_pb2_grpc

class GRPCTransport(DataTransport):
    def __init__(self, config: Dict):
        self.config = config
        self.channel = None
        self.stub = None

    async def connect(self):
        self.channel = grpc.aio.insecure_channel(self.config['address'])
        self.stub = tts_pb2_grpc.TTSServiceStub(self.channel)

    async def send(self, data: Any, target: Optional[str] = None) -> bool:
        if not self.stub:
            await self.connect()

        request = tts_pb2.InferenceRequest(
            data=pickle.dumps(data)
        )
        await self.stub.Inference(request)
        return True
```

#### 1.2 配置化部署

```yaml
# config/distributed.yaml
deployment:
  mode: "distributed"
  transport: "grpc"  # grpc | http | zmq

services:
  encode:
    enabled: true
    replicas: 2
    host: "encode-service.default.svc.cluster.local"
    port: 8081
    resources:
      cpu: "2"
      memory: "4Gi"

  llm:
    enabled: true
    replicas: 1
    host: "llm-service.default.svc.cluster.local"
    port: 8082
    resources:
      cpu: "8"
      memory: "32Gi"
      nvidia.com/gpu: "1"
    model_path: "/models/CosyVoice2-0.5B"
    config:
      max_total_token_num: 65536
      batch_max_tokens: 8192

  decode:
    enabled: true
    replicas: 2
    host: "decode-service.default.svc.cluster.local"
    port: 8083
    resources:
      cpu: "4"
      memory: "16Gi"
      nvidia.com/gpu: "1"
    config:
      decode_max_batch_size: 4
      load_trt: true

cache:
  type: "redis"
  address: "redis.cache.svc.cluster.local:6379"
  ttl: 3600
  max_memory: "8Gi"

monitoring:
  prometheus: "http://prometheus.monitoring.svc.cluster.local:9090"
  jaeger: "http://jaeger.monitoring.svc.cluster.local:16686"
```

#### 1.3 分布式缓存实现

```python
# light_tts/server/cache/distributed_cache.py
import redis.asyncio as redis
import pickle
from typing import Any, Optional

class DistributedCache:
    def __init__(self, config: Dict):
        self.redis_client = redis.Redis(
            host=config['host'],
            port=config['port'],
            decode_responses=False,
            max_connections=config.get('max_connections', 100)
        )
        self.ttl = config.get('ttl', 3600)

    async def set(self, key: str, value: Any) -> bool:
        try:
            serialized = pickle.dumps(value)
            await self.redis_client.setex(key, self.ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    async def get(self, key: str) -> Optional[Any]:
        try:
            value = await self.redis_client.get(key)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

# light_tts/server/shm_tools/distributed_speech_manager.py
from .distributed_cache import DistributedCache

class DistributedSpeechManager:
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.cache = DistributedCache(config)
        self.local_cache = {}
        self.max_local_size = config.get('max_local_size', 100)

    async def store_speech_embedding(self, speech_index: int, embedding: np.ndarray):
        """存储说话人嵌入"""
        key = f"{self.name}:embedding:{speech_index}"
        await self.cache.set(key, embedding)

        # 本地缓存热数据
        if len(self.local_cache) < self.max_local_size:
            self.local_cache[speech_index] = embedding

    async def get_speech_embedding(self, speech_index: int) -> Optional[np.ndarray]:
        """获取说话人嵌入"""
        # 先查本地缓存
        if speech_index in self.local_cache:
            return self.local_cache[speech_index]

        # 查分布式缓存
        key = f"{self.name}:embedding:{speech_index}"
        embedding = await self.cache.get(key)
        if embedding is not None:
            self.local_cache[speech_index] = embedding
        return embedding
```

### 第二阶段：服务部署

#### 2.1 Docker 镜像构建

```dockerfile
# Dockerfile.llm
FROM nvidia/cuda:11.8-devel-ubuntu20.04

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 复制代码
COPY light_tts/ ./light_tts/
COPY cosyvoice/ ./cosyvoice/

# 启动脚本
COPY scripts/start_llm.sh /start_llm.sh
RUN chmod +x /start_llm.sh

EXPOSE 8082

CMD ["/start_llm.sh"]
```

```bash
#!/bin/bash
# scripts/start_llm.sh

# 加载配置
python -c "
import yaml
with open('/config/distributed.yaml') as f:
    config = yaml.safe_load(f)

import json
with open('/tmp/llm_config.json', 'w') as f:
    json.dump(config['services']['llm']['config'], f)
"

# 启动 LLM 服务
python -m light_tts.server.llm_service \
  --config /tmp/llm_config.json \
  --transport grpc \
  --port 8082
```

#### 2.2 Kubernetes 部署

```yaml
# k8s/llm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-service
spec:
  replicas: 1
  selector:
    matchLabels:
      app: llm-service
  template:
    metadata:
      labels:
        app: llm-service
    spec:
      containers:
      - name: llm
        image: light-tts/llm:latest
        ports:
        - containerPort: 8082
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "24Gi"
            cpu: "8"
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "16"
        volumeMounts:
        - name: model-storage
          mountPath: /models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: llm-service
spec:
  selector:
    app: llm-service
  ports:
  - port: 8082
    targetPort: 8082
  type: ClusterIP
```

#### 2.3 服务发现与负载均衡

```python
# light_tts/server/discovery/service_discovery.py
import aiohttp
import asyncio
from typing import List, Dict

class ServiceDiscovery:
    def __init__(self, config: Dict):
        self.config = config
        self.services = {}
        self.health_check_interval = config.get('health_check_interval', 30)

    async def discover_services(self):
        """发现可用服务"""
        for service_name, service_config in self.config['services'].items():
            if not service_config.get('enabled', True):
                continue

            # 从 Kubernetes API 或 Consul 获取服务实例
            instances = await self._get_service_instances(
                service_config['host'],
                service_config['port']
            )

            self.services[service_name] = {
                'instances': instances,
                'current_index': 0
            }

    async def get_service_endpoint(self, service_name: str) -> str:
        """获取服务端点（负载均衡）"""
        if service_name not in self.services:
            await self.discover_services()

        service = self.services[service_name]
        instances = service['instances']

        if not instances:
            raise Exception(f"No available instances for {service_name}")

        # 轮询负载均衡
        endpoint = instances[service['current_index']]
        service['current_index'] = (service['current_index'] + 1) % len(instances)

        return endpoint

    async def health_check(self):
        """健康检查"""
        while True:
            for service_name, service in self.services.items():
                healthy_instances = []
                for instance in service['instances']:
                    if await self._check_instance_health(instance):
                        healthy_instances.append(instance)
                service['instances'] = healthy_instances

            await asyncio.sleep(self.health_check_interval)
```

### 第三阶段：性能优化

#### 3.1 连接池管理

```python
# light_tts/server/transport/connection_pool.py
import aiohttp
import asyncio
from typing import Dict, List

class ConnectionPool:
    def __init__(self, config: Dict):
        self.config = config
        self.pools = {}

    async def get_pool(self, service: str) -> aiohttp.ClientSession:
        if service not in self.pools:
            connector = aiohttp.TCPConnector(
                limit=self.config.get('max_connections', 100),
                limit_per_host=self.config.get('max_per_host', 10),
                ttl_dns_cache=300,
                use_dns_cache=True,
            )

            timeout = aiohttp.ClientTimeout(
                total=self.config.get('timeout', 30),
                connect=self.config.get('connect_timeout', 5),
                sock_connect=self.config.get('sock_connect_timeout', 5),
                sock_read=self.config.get('sock_read_timeout', 10)
            )

            self.pools[service] = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )

        return self.pools[service]
```

#### 3.2 批处理优化

```python
# light_tts/server/llm/batch_optimizer.py
import asyncio
from collections import defaultdict
from typing import List, Dict

class DistributedBatchOptimizer:
    def __init__(self, config: Dict):
        self.config = config
        self.pending_requests = defaultdict(list)
        self.batch_size = config.get('batch_size', 8)
        self.max_wait_time = config.get('max_wait_time', 0.05)  # 50ms

    async def add_request(self, request: Dict):
        """添加请求到批处理队列"""
        service = request['target_service']
        self.pending_requests[service].append(request)

        # 检查是否需要触发批处理
        if len(self.pending_requests[service]) >= self.batch_size:
            await self._process_batch(service)
        else:
            # 设置超时处理
            asyncio.create_task(self._timeout_process(service))

    async def _process_batch(self, service: str):
        """处理批请求"""
        if not self.pending_requests[service]:
            return

        batch = self.pending_requests[service][:self.batch_size]
        self.pending_requests[service] = self.pending_requests[service][self.batch_size:]

        # 发送批请求
        transport = await self._get_transport(service)
        await transport.send_batch(batch)
```

## 监控与运维

### 监控指标

```python
# light_tts/server/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# 业务指标
REQUEST_COUNT = Counter('light_tts_requests_total', 'Total requests', ['service', 'method'])
REQUEST_DURATION = Histogram('light_tts_request_duration_seconds', 'Request duration', ['service'])
BATCH_SIZE = Histogram('light_tts_batch_size', 'Batch processing size', ['service'])
QUEUE_LENGTH = Gauge('light_tts_queue_length', 'Queue length', ['service'])

# 系统指标
GPU_UTILIZATION = Gauge('light_tts_gpu_utilization', 'GPU utilization', ['gpu_id'])
MEMORY_USAGE = Gauge('light_tts_memory_usage', 'Memory usage', ['type'])
CACHE_HIT_RATE = Gauge('light_tts_cache_hit_rate', 'Cache hit rate', ['cache_type'])
```

### 日志配置

```yaml
# config/logging.yaml
version: 1
formatters:
  json:
    format: '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "service": "%(name)s", "message": "%(message)s"}'
  standard:
    format: '%(asctime)s [%(levelname)s] %(name)s: %(message)s'

handlers:
  file:
    class: logging.handlers.RotatingFileHandler
    filename: /var/log/light-tts/app.log
    maxBytes: 104857600  # 100MB
    backupCount: 10
    formatter: json

  console:
    class: logging.StreamHandler
    formatter: standard

loggers:
  light_tts:
    level: INFO
    handlers: [file, console]
    propagate: false

root:
  level: INFO
  handlers: [file, console]
```

## 故障处理

### 熔断机制

```python
# light_tts/server/resilience/circuit_breaker.py
import asyncio
from enum import Enum
from typing import Callable, Any

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.next_attempt = None

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        if self.state == CircuitState.OPEN:
            if asyncio.get_event_loop().time() < self.next_attempt:
                raise Exception("Circuit breaker is OPEN")
            self.state = CircuitState.HALF_OPEN

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.next_attempt = asyncio.get_event_loop().time() + self.timeout
```

## 部署清单

### 生产环境检查清单

- [ ] 服务拆分完成，各模块独立部署
- [ ] 网络传输层实现（gRPC/HTTP/ZMQ）
- [ ] 分布式缓存配置（Redis）
- [ ] 负载均衡器配置
- [ ] 监控系统部署
- [ ] 日志聚合配置
- [ ] 熔断和重试机制
- [ ] 健康检查配置
- [ ] 自动扩缩容策略
- [ ] 备份和恢复方案

### 性能基准

| 指标 | 目标值 | 监控方式 |
|------|--------|----------|
| P99 延迟 | < 500ms | Prometheus + Grafana |
| 吞吐量 | > 100 RPS | 压力测试 |
| 可用性 | 99.9% | 健康检查 |
| GPU 利用率 | > 70% | NVIDIA DCGM |
| 缓存命中率 | > 80% | Redis 监控 |

## 总结

分布式分离部署方案提供了更好的可扩展性和资源利用率，适合大规模生产环境。通过合理的架构设计和性能优化，可以实现高并发、低延迟的 TTS 服务。