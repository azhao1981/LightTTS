# Light-TTS decode_process_num 限制分析与修复

## 🚨 问题概述

在 `light_tts/server/api_start.py` 中发现了一个关键问题：

```python
# 第111-112行
num_loras = 1
assert args.decode_process_num <= num_loras
```

这个限制导致 `decode_process_num` 最大只能为 1，与文档中的性能优化建议相矛盾。

## 📊 问题分析

### 1. 当前实现的逻辑

#### 端口分配逻辑
```python
# api_start.py 第114-143行
num_loras = 1  # 硬编码为1，表示只有1个LoRA模型

# LLM端口分配：每个模型风格1个端口
tts_llm_ports = can_use_ports[0 : num_loras]  # 只有1个端口

# Decode端口分配：也是每个模型风格1个端口
tts_decode_ports = can_use_ports[0 : num_loras]  # 只有1个端口！
```

#### 进程启动逻辑
```python
# api_start.py 第157-163行
for decode_proc_index in range(args.decode_process_num):  # 尝试启动多个进程
    for style_name, tts_decode_port in zip(["CosyVoice2"], tts_decode_ports):  # 但只有1个端口
        tmp_args.append((args, tts_decode_port, httpserver_port, style_name, decode_parall_lock, decode_proc_index))
```

### 2. 设计意图 vs 实际实现

| 概念 | 设计意图 | 实际实现 | 问题 |
|------|----------|----------|------|
| LoRA | 多个语音风格模型 | 只有CosyVoice2一个模型 | num_loras硬编码为1 |
| decode_process_num | Decode worker进程数 | 被错误限制为≤1 | 与LoRA概念混淆 |
| 端口分配 | 每个模型1个端口 | 多进程共享1个端口 | 端口冲突风险 |

### 3. 文档与代码的矛盾

**文档中的建议** (docs/performance-tuning.md):
```
| RTX 4090 | 24GB | 3 workers, decode_process_num=2 | 最佳性能 |
| RTX 3090 | 24GB | 3 workers, decode_process_num=2 | 高性能 |
| A100 | 40GB | 4 workers, decode_process_num=3 | 最高性能 |
```

**代码中的限制**:
```python
assert args.decode_process_num <= num_loras  # num_loras = 1
```

**结果**: 文档建议使用2-3个decode workers，但代码只允许1个！

## 🔧 修复方案

### 方案1：最小改动 - 修复端口分配（推荐）

#### 修改端口分配逻辑
```python
# api_start.py 修改后的代码
num_loras = 1
# 移除错误的限制
# assert args.decode_process_num <= num_loras

# 为每个decode进程分配独立端口
tts_decode_ports = can_use_ports[0 : args.decode_process_num]
del can_use_ports[0 : args.decode_process_num]
```

#### 使用ZMQ的PUSH-PULL负载均衡
```python
# LLM Manager (tts_llm/manager.py) 修改
def __init__(self, ...):
    # 为每个decode端口创建独立的socket
    self.decode_sockets = []
    for port in tts_decode_ports:
        socket = context.socket(zmq.PUSH)
        socket.connect(f"{args.zmq_mode}127.0.0.1:{port}")
        self.decode_sockets.append(socket)
        # 使用轮询负载均衡
        self.current_decode_socket = 0

def _send_to_tts2_decodec_proc(self, batch: Batch):
    for req in batch.reqs:
        if req.finish_status.is_finished():
            # 轮询发送到不同的decode进程
            socket = self.decode_sockets[self.current_decode_socket]
            self.current_decode_socket = (self.current_decode_socket + 1) % len(self.decode_sockets)

            socket.send_pyobj((req.request_id, req.get_output_len()), protocol=pickle.HIGHEST_PROTOCOL)
```

### 方案2：使用ZMQ Proxy（更优雅）

#### 添加ZMQ Proxy作为负载均衡器
```python
# 新建文件：light_tts/server/zmq_proxy.py
import zmq
import threading

class ZMQProxy:
    def __init__(self, frontend_port: int, backend_ports: list):
        self.context = zmq.Context()

        # LLM连接到frontend
        self.frontend = self.context.socket(zmq.PULL)
        self.frontend.bind(f"tcp://*:{frontend_port}")

        # Decode workers连接到backend
        self.backend = self.context.socket(zmq.PUSH)
        for port in backend_ports:
            self.backend.bind(f"tcp://*:{port}")

        self.running = False

    def start(self):
        """启动proxy"""
        self.running = True
        threading.Thread(target=self._proxy_loop, daemon=True).start()

    def _proxy_loop(self):
        """消息转发循环"""
        while self.running:
            try:
                # 从LLM接收
                message = self.frontend.recv_pyobj()
                # 转发到可用的Decode worker
                self.backend.send_pyobj(message)
            except Exception as e:
                print(f"Proxy error: {e}")

    def stop(self):
        self.running = False
```

#### 修改启动脚本
```python
# api_start.py
from light_tts.server.zmq_proxy import ZMQProxy

# 在启动decode workers之前启动proxy
proxy_frontend_port = can_use_ports[0]
decode_backend_ports = can_use_ports[1:1+args.decode_process_num]
del can_use_ports[0:1+args.decode_process_num]

# 启动proxy
proxy = ZMQProxy(proxy_frontend_port, decode_backend_ports)
proxy.start()

# LLM连接到proxy frontend
tts_llm_to_decode_port = proxy_frontend_port

# Decode workers使用各自的backend端口
```

### 方案3：完整重构（长期方案）

#### 分离LoRA和worker概念
```python
# 新配置结构
config = {
    "models": [
        {
            "name": "CosyVoice2",
            "path": "/models/CosyVoice2",
            "workers": 3  # 每个模型的worker数
        }
    ],
    "load_balancing": "round_robin"  # 负载均衡策略
}
```

## ⚠️ 去掉限制的潜在问题

### 1. 端口冲突
**问题**: 多个进程尝试绑定相同端口
```bash
# 错误示例
Decode Process 1: bind(0.0.0.0:8083) ✓
Decode Process 2: bind(0.0.0.0:8083) ✗ Address already in use
```

### 2. ZMQ连接问题
**问题**: 多个PUSH连接到同一个PULL可能导致消息丢失
```python
# 错误的连接方式
LLM -> PUSH -> tcp://localhost:8083
Decode1 -> PULL <- tcp://localhost:8083  # 可以接收
Decode2 -> PULL <- tcp://localhost:8083  # 消息可能被Decode1接收
```

### 3. 负载不均
**问题**: 消息可能被单个worker接收，其他worker空闲

## 🧪 测试验证

### 验证脚本
```python
# test/verify_multi_decode.py
import asyncio
import aiohttp
import time
import statistics

async def test_multi_decode():
    """测试多decode worker是否正常工作"""

    # 测试参数
    concurrent_requests = 20
    test_text = "这是一个测试文本，用于验证多worker是否正常工作。"

    async with aiohttp.ClientSession() as session:
        # 发送并发请求
        tasks = []
        for i in range(concurrent_requests):
            task = send_request(session, test_text, i)
            tasks.append(task)

        # 收集结果
        latencies = await asyncio.gather(*tasks)

        # 分析结果
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile

        print(f"并发请求: {concurrent_requests}")
        print(f"平均延迟: {avg_latency:.2f}ms")
        print(f"P95延迟: {p95_latency:.2f}ms")

        # 检查GPU利用率
        print("\n检查GPU利用率...")
        print("如果有多个decode workers，应该看到多个GPU进程")

async def send_request(session, text, request_id):
    """发送单个请求"""
    start_time = time.time()

    # 构造请求
    files = {
        "prompt_wav": ("sample.wav", open("../cosyvoice/asset/zero_shot_prompt.wav", "rb"), "audio/wav")
    }
    data = {
        "tts_text": text,
        "prompt_text": "希望你以后能够做的比我还好呦。",
    }

    async with session.post("http://localhost:8080/inference_zero_shot", files=files, data=data) as resp:
        await resp.read()
        return (time.time() - start_time) * 1000

if __name__ == "__main__":
    asyncio.run(test_multi_decode())
```

### 监控脚本
```bash
#!/bin/bash
# monitor_decode_workers.sh

echo "监控Decode Workers..."
echo "================================"

# 检查进程数
echo "1. 检查进程数:"
ps aux | grep "tts_decode" | grep -v grep | wc -l | xargs echo "  Decode进程数:"

# 检查端口使用
echo -e "\n2. 检查端口使用:"
netstat -tlnp 2>/dev/null | grep :808 | head -10

# 检查GPU进程
echo -e "\n3. 检查GPU进程:"
nvidia-smi pmon -s u -c 1

# 检查日志
echo -e "\n4. 最近的错误日志:"
tail -n 20 logs/light-tts.log 2>/dev/null | grep -i error || echo "  无错误日志"
```

## 📋 修复检查清单

### 立即修复（紧急）
- [ ] 移除 `assert args.decode_process_num <= num_loras`
- [ ] 修改端口分配逻辑，为每个decode进程分配独立端口
- [ ] 更新LLM到Decode的发送逻辑，支持多端口轮询

### 短期优化（1周内）
- [ ] 实现ZMQ Proxy负载均衡器
- [ ] 添加decode worker健康检查
- [ ] 修复文档与代码的不一致

### 长期重构（1个月内）
- [ ] 分离LoRA模型和worker概念
- [ ] 实现动态负载均衡策略
- [ ] 添加自动扩缩容支持

## 🎯 推荐的修复步骤

### 第一步：立即修复（5分钟）
```python
# 修改 api_start.py 第142-143行
# 原代码：
tts_decode_ports = can_use_ports[0 : num_loras]
del can_use_ports[0 : num_loras]

# 修改为：
tts_decode_ports = can_use_ports[0 : args.decode_process_num]
del can_use_ports[0 : args.decode_process_num]
```

### 第二步：修复LLM发送逻辑（30分钟）
在 `tts_llm/manager.py` 的 `_send_to_tts2_decodec_proc` 方法中实现轮询发送。

### 第三步：测试验证（10分钟）
运行验证脚本，确保多worker正常工作。

### 第四步：更新文档（5分钟）
修正 `docs/performance-tuning.md` 中的说明。

## 📝 总结

这个限制是一个**历史遗留问题**，原本设计支持多LoRA模型，但现在实际需求是单模型多workers。修复这个问题对于发挥多GPU性能至关重要。

**修复后可以实现的性能提升**：
- RTX 4090: 从1个worker → 3个workers，性能提升2-3倍
- A100: 从1个worker → 4个workers，性能提升3-4倍
- 充分利用所有GPU资源，避免GPU闲置

**影响范围**：
- 仅需修改3个文件
- 不影响API接口
- 向后兼容

这是一个**高优先级**的修复，建议立即实施。