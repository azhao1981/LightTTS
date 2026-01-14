# Encode 模块处理代码详解

## 目录结构

```
light_tts/server/tts_encode/
├── __init__.py
├── manager.py                     ⭐ Encode Manager 核心逻辑
└── model_infer/
    ├── __init__.py
    └── frontend.py                # CosyVoice Frontend 封装
```

---

## 核心文件：manager.py

**文件路径**: `light_tts/server/tts_encode/manager.py`

### 类定义：TTS1EncodeManager

```python
class TTS1EncodeManager:
    def __init__(self, args, tts_llm_ports, tts1_encode_port, index_id, encode_parall_lock):
        # 初始化 ZMQ 通信
        self.recv_from_httpserver = context.socket(zmq.PULL)
        self.recv_from_httpserver.bind(f"{args.zmq_mode}127.0.0.1:{tts1_encode_port}")

        # 连接到 LLM 模块（每个进程负责部分 lora）
        self.send_to_tts_llms = {}
        for i, lora_w in enumerate(self.model_cfg["lora_info"]):
            if i % args.encode_process_num == self.index_id:
                send_to_tts_llm_i = context.socket(zmq.PUSH)
                send_to_tts_llm_i.connect(f"{args.zmq_mode}127.0.0.1:{tts_llm_ports[i]}")
                self.send_to_tts_llms[lora_w["style_name"]] = send_to_tts_llm_i

        # 初始化 Frontend
        self.frontend = CosyVoiceFrontEnd(
            configs['get_tokenizer'],
            configs['feat_extractor'],
            '{}/campplus.onnx'.format(args.model_dir),
            '{}/speech_tokenizer_v2.onnx'.format(args.model_dir),
            '{}/spk2info.pt'.format(args.model_dir),
            configs['allowed_special']
        )

        # 初始化共享内存
        self.shared_speech_manager = SharedSpeechManager(f"{args.port}_cosyvoice", args.cache_capacity)
        self.shm_req_manager = ShmReqManager()

        # LLM 配置（用于 vocab offset）
        self.model_config = configs["llm"].llm.model.model.config
        self.vocab_size = self.model_config.vocab_size  # ⭐ 关键配置
```

---

## 主处理循环：loop_for_fwd()

**位置**: `manager.py:81-182`

这是 Encode 模块的核心处理逻辑。

### 整体流程

```python
async def loop_for_fwd(self):
    module_name = "tts1_encoder"
    idle_count = 0
    while True:
        # 空闲时等待
        if len(self.waiting_reqs) == 0:
            await asyncio.sleep(0.01)
        else:
            # 处理请求队列
            n = len(self.waiting_reqs)
            while n > 0:
                req = self.waiting_reqs.pop(0)

                # 检查 SFT 模式
                spk_id = getattr(req, 'spk_id', '')
                is_sft = spk_id.endswith('_sft') if spk_id else False

                if is_sft:
                    # 📍 SFT 模式分支 (第 110-150 行)
                    ...
                else:
                    # 📍 Zero-Shot 模式分支 (第 152-182 行)
                    ...
```

---

## 分支 1：SFT 模式处理

**代码位置**: `manager.py:110-150`

### 流程图

```
if is_sft:
    ├─ 1. 等待共享内存数据准备
    │    if not self.shared_speech_manager.speech_data_ready(speech_index):
    │        self.waiting_reqs.append(req)  # 重新加入队列
    │        continue
    │
    ├─ 2. 从共享内存读取数据
    │    speech_data = self.shared_speech_manager.get_index_speech(speech_index)
    │    speech_token, speech_feat, embedding = speech_data
    │
    ├─ 3. ⭐ 关键修复：添加 vocab offset (v1.0.1)
    │    if speech_token.size > 0:
    │        speech_token_offset = (speech_token + self.vocab_size + 2)
    │        audio_ids = speech_token_offset.flatten().tolist()
    │
    ├─ 4. 计算 prompt_token_pad
    │    req.prompt_token_pad = ...
    │
    └─ 5. 发送到 LLM
         self.send_to_tts_llms[tts_model_name].send_pyobj(req.index_in_shm_mem)
```

### 详细代码

```python
# 第 110-150 行
if is_sft:
    base_spk_id = spk_id[:-4]

    # 等待 API 层存储的 embedding 数据
    if not self.shared_speech_manager.speech_data_ready(speech_index):
        self.waiting_reqs.append(req)
        continue

    # 从共享内存读取
    speech_data = self.shared_speech_manager.get_index_speech(speech_index)
    if speech_data is None:
        logger.error(f"SFT mode: speech_index {speech_index} data not ready")
        req.router_aborted = True
        self.shm_req_manager.put_back_req_obj(req)
        req.can_released_mark = True
        continue

    speech_token, speech_feat, embedding = speech_data
    logger.info(f"SFT mode: req_id {req.request_id}, spk_id={spk_id}, embedding shape={embedding.shape}")

    # ⭐ v1.0.1 修复：添加 vocab offset
    if not req.bistream:
        if speech_token.size > 0:
            # Add vocab_size + 2 offset for correct embedding lookup
            speech_token_offset = (speech_token + self.vocab_size + 2)
            audio_ids = speech_token_offset.flatten().tolist()
        else:
            audio_ids = []
        with self.shm_req_manager.get_req_lock_by_index(req.index_in_shm_mem):
            req.set_speech_token(audio_ids)

    # 计算 padding
    req.prompt_token_pad = int(np.ceil(speech_token.size / self.token_hop_len) * self.token_hop_len - speech_token.size) if speech_token.size > 0 else 0

    # 发送到 LLM
    self.shm_req_manager.put_back_req_obj(req)
    self.send_to_tts_llms[tts_model_name].send_pyobj(req.index_in_shm_mem)
```

---

## 分支 2：Zero-Shot 模式处理

**代码位置**: `manager.py:152-182`

### 流程图

```
if need_extract_speech:
    ├─ 1. 从共享内存读取原始音频
    │    prompt_speech_16k = self.shared_speech_manager.get_index_data(speech_index)
    │
    ├─ 2. ⭐ 调用 frontend_zero_shot 提取特征
    │    model_input = self.frontend.frontend_zero_shot(
    │        '', '', prompt_speech_16k, self.resample_rate, ''
    │    )
    │
    ├─ 3. 提取特征
    │    speech_token = model_input["llm_prompt_speech_token"]
    │    speech_feat = model_input["prompt_speech_feat"]
    │    embedding = model_input["llm_embedding"]
    │
    ├─ 4. 存储到共享内存（供后续使用）
    │    self.shared_speech_manager.set_index_speech(speech_index, ...)
    │
else:  # 使用缓存的特征
    ├─ 5. 从共享内存读取缓存的 speech_token
    │    speech_token = self.shared_speech_manager.get_index_speech_token(speech_index)
    │
├─ 6. ⭐ 添加 vocab offset
│    speech_token = (speech_token + self.vocab_size + 2)
│
├─ 7. 设置到 req 对象
│    req.set_speech_token(audio_ids)
│
└─ 8. 发送到 LLM
     self.send_to_tts_llms[tts_model_name].send_pyobj(req.index_in_shm_mem)
```

### 详细代码

```python
# 第 152-182 行
if need_extract_speech:
    # 首次使用：提取特征
    logger.debug(f"tts_encode req_id {req.request_id} generate speech index {speech_index} cache")
    prompt_speech_16k = self.shared_speech_manager.get_index_data(speech_index)
    if prompt_speech_16k is None:
        raise RuntimeError(f"In encode, get_index_data {speech_index} not found")
    prompt_speech_16k = torch.from_numpy(prompt_speech_16k.arr)

    # ⭐ 调用 frontend_zero_shot
    model_input = self.frontend.frontend_zero_shot('', '', prompt_speech_16k, self.resample_rate, '')
    speech_token = model_input["llm_prompt_speech_token"].cpu().numpy()
    speech_feat = model_input["prompt_speech_feat"].squeeze(0).cpu().numpy()
    embedding = model_input["llm_embedding"].cpu().numpy()

    # 存储到共享内存
    self.shared_speech_manager.set_index_speech(speech_index, speech_token, speech_feat, embedding)
else:
    # 使用缓存的特征
    if not self.shared_speech_manager.speech_data_ready(speech_index):
        self.waiting_reqs.append(req)
        continue
    else:
        logger.debug(f"tts_encode req_id {req.request_id} use speech index {speech_index} cache")
        speech_token = self.shared_speech_manager.get_index_speech_token(speech_index).arr[0]

# ⭐ 添加 vocab offset（两种模式都需要）
if not req.bistream:
    speech_token = (speech_token + self.vocab_size + 2)
    audio_ids = speech_token.flatten().tolist()
    with self.shm_req_manager.get_req_lock_by_index(req.index_in_shm_mem):
        req.set_speech_token(audio_ids)

req.prompt_token_pad = int(np.ceil(speech_token.size / self.token_hop_len) * self.token_hop_len - speech_token.size)

# 发送到 LLM
self.shm_req_manager.put_back_req_obj(req)
self.send_to_tts_llms[tts_model_name].send_pyobj(req.index_in_shm_mem)
```

---

## 网络通信循环：loop_for_netio_req()

**代码位置**: `manager.py:184-191`

```python
async def loop_for_netio_req(self):
    while True:
        # 接收来自 HttpServerManager 的请求
        recv_req = await self.recv_from_httpserver.recv_pyobj()

        if isinstance(recv_req, GroupReqIndexes):
            self.add_req(recv_req)
        else:
            assert False, f"Error Req Inf {recv_req}"
```

---

## 进程启动流程

### 1. 启动位置

**文件**: `light_tts/server/api_start.py:137-140`

```python
encode_parall_lock = mp.Semaphore(args.encode_paral_num)
for index_id in range(args.encode_process_num):
    funcs.append(start_tts1_encode_process)
    start_args.append((
        args,
        tts_llm_ports,
        tts1_encode_ports[index_id],  # 每个 Encode 进程有独立端口
        index_id,                      # 进程 ID
        encode_parall_lock
    ))

# 启动第一个 Encode 进程（先启动一个解决同步问题）
process_manager.start_submodule_processes(start_funcs=funcs[0:1], start_args=start_args[0:1])
```

### 2. 进程入口函数

**文件**: `light_tts/server/tts_encode/manager.py:197-227`

```python
def start_tts1_encode_process(args, tts_llm_ports, tts1_encode_port, index_id, encode_parall_lock, pipe_writer):
    # 注册 graceful 退出处理
    graceful_registry(inspect.currentframe().f_code.co_name)
    start_parent_check_thread()

    try:
        # 创建 Encode Manager
        encodeserver = TTS1EncodeManager(
            args,
            tts_llm_ports,
            tts1_encode_port,
            index_id,
            encode_parall_lock
        )
    except Exception as e:
        # 错误处理
        pipe_writer.send(err_str)
        encodeserver.clean_up()
        raise

    pipe_writer.send('init ok')
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # 启动两个协程
    loop.create_task(encodeserver.loop_for_fwd())       # 处理请求
    loop.run_until_complete(encodeserver.loop_for_netio_req())  # 接收请求
```

### 3. 事件循环

```python
# 两个异步任务并发执行
loop.create_task(encodeserver.loop_for_fwd())           # 任务1：处理队列中的请求
loop.run_until_complete(encodeserver.loop_for_netio_req())  # 任务2：接收网络请求
```

---

## ZMQ 通信架构

### Encode 进程的 ZMQ Socket

| Socket | 类型 | 连接 | 用途 |
|--------|------|------|------|
| `recv_from_httpserver` | PULL | bind to `*:tts1_encode_port` | 接收 HttpServerManager 的请求 |
| `send_to_tts_llms[style]` | PUSH | connect to `127.0.0.1:tts_llm_port` | 发送处理后的请求到 LLM |

### 数据流向

```
┌─────────────────────┐
│ HttpServerManager   │
│                     │
│ send_to_tts1_encode │─── ZMQ PUSH ────┐
│   [style]           │                  │
└─────────────────────┘                  │
                                         │
                                         ↓
┌──────────────────────────────────────────────────┐
│         Encode Process (多个进程)                 │
│                                                  │
│  recv_from_httpserver ← ZMQ PULL (bind)          │
│          ↓                                       │
│  add_req → waiting_reqs                          │
│          ↓                                       │
│  loop_for_fwd()                                  │
│     ├─ SFT 分支 (110-150)                        │
│     └─ Zero-Shot 分支 (152-182)                  │
│          ↓                                       │
│  send_to_tts_llms[style] ── ZMQ PUSH ────→      │
└──────────────────────────────────────────────────┘
```

---

## 关键参数配置

### CLI 参数

**文件**: `light_tts/server/api_cli.py:29`

```bash
--encode_process_num       # Encode 进程数量 (默认: 1)
--encode_paral_num         # 每个 Encode 进程的并行度 (默认: ?)
```

### Semaphore 控制

```python
encode_parall_lock = mp.Semaphore(args.encode_paral_num)
```

这个 Semaphore 用于控制多个 Encode 进程之间的并发，避免同时处理过多请求。

---

## SFT vs Zero-Shot 对比

| 项目 | SFT 模式 | Zero-Shot 模式 |
|------|---------|----------------|
| **分支代码** | 110-150 行 | 152-182 行 |
| **检测方式** | `spk_id.endswith('_sft')` | `need_extract_speech` |
| **数据来源** | 共享内存（API 层存储） | 共享内存或实时提取 |
| **调用 frontend** | ❌ 不调用 | ✅ 调用 frontend_zero_shot |
| **vocab offset** | ✅ 第 133-136 行 | ✅ 第 172 行 |
| **特征提取** | API 层已完成 | Encode 层提取或使用缓存 |

---

## 关键修复点 (v1.0.1)

### 1. SFT 模式的 vocab offset (第 133-136 行)

```python
# 修复前（不存在）
# SFT 分支直接跳过了 vocab offset 处理

# 修复后
if speech_token.size > 0:
    speech_token_offset = (speech_token + self.vocab_size + 2)
    audio_ids = speech_token_offset.flatten().tolist()
else:
    audio_ids = []
```

### 2. SFT 模式的 prompt_token_pad (第 142 行)

```python
# 修复前
req.prompt_token_pad = 0

# 修复后
req.prompt_token_pad = int(np.ceil(speech_token.size / self.token_hop_len) * self.token_hop_len - speech_token.size) if speech_token.size > 0 else 0
```

---

## 性能优化点

### 1. 并行处理

- **多进程**: `--encode_process_num` 控制进程数
- **每进程并行**: Semaphore 控制并发度
- **Lora 分片**: 每个 Encode 进程处理部分 lora

```python
# 第 43-48 行
for i, lora_w in enumerate(self.model_cfg["lora_info"]):
    if i % args.encode_process_num == self.index_id:  # 分片分配
        send_to_tts_llm_i = context.socket(zmq.PUSH)
        send_to_tts_llm_i.connect(f"{args.zmq_mode}127.0.0.1:{tts_llm_ports[i]}")
        self.send_to_tts_llms[lora_w["style_name"]] = send_to_tts_llm_i
```

### 2. 共享内存缓存

- **Zero-Shot**: 首次提取后缓存 `speech_token`, `speech_feat`, `embedding`
- **SFT**: API 层预先存储，Encode 层直接使用

### 3. 等待队列

```python
# 第 115-117 行
if not self.shared_speech_manager.speech_data_ready(speech_index):
    self.waiting_reqs.append(req)  # 重新加入队列
    continue
```

避免忙等待，将请求放回队列稍后处理。

---

## 相关文档

- **完整模块指南**：
  - [DECODE_MODULE_GUIDE.md](DECODE_MODULE_GUIDE.md) - Decode 模块详解 ⭐
- [SFT_MODE.md](SFT_MODE.md) - SFT 模式使用文档
- [SFT_MODE_DATA_FLOW.md](SFT_MODE_DATA_FLOW.md) - 完整数据流程图
- [FRONTEND_CALL_CHAIN.md](FRONTEND_CALL_CHAIN.md) - Frontend 调用链分析
- [sft_bug.md](sft_bug.md) - Bug 分析与修复记录
