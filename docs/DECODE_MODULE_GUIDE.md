# Decode 模块处理代码详解

## 目录结构

```
light_tts/server/tts_decode/
├── __init__.py
├── manager.py                     ⭐ Decode Manager 核心逻辑
├── decode_req.py                  # Decode 请求封装
└── model_infer/
    ├── __init__.py
    ├── model_rpc.py               ⭐ 模型推理 RPC 服务
    └── patch_conditional_cfm.py   # Flow 模型条件补丁
```

---

## 核心文件：model_rpc.py

**文件路径**: `light_tts/server/tts_decode/model_infer/model_rpc.py`

### 类定义：TTS2DecodeModelRpcServer

```python
class TTS2DecodeModelRpcServer():
    def init_model(self, kvargs):
        model_dir = kvargs["model_dir"]
        configs = load_yaml(model_dir)

        # 创建 CosyVoice2 模型（只包含 flow 和 hift）
        self.model = CosyVoice2Model(
            configs['llm'],    # 配置，但不加载 llm
            configs['flow'],   # Flow 模型配置
            configs['hift'],   # HiFi-GAN 模型配置
            fp16=True
        )

        # 加载模型权重
        self.model.load(
            '{}/llm.pt'.format(model_dir),     # LLM 权重（立即删除）
            '{}/flow.pt'.format(model_dir),    # ⭐ Flow 模型权重
            '{}/hift.pt'.format(model_dir)     # ⭐ HiFi-GAN 模型权重
        )

        del self.model.llm  # Decode 不需要 LLM，删除以节省内存

        # 可选：加载加速版本
        load_jit = kvargs.get("load_jit", False)
        load_trt = kvargs.get("load_trt", False)

        if load_jit:
            self.model.load_jit('{}/flow.encoder.{}.zip'.format(
                model_dir, 'fp16' if self.fp16 else 'fp32'
            ))

        if load_trt:
            capability = torch.cuda.get_device_capability(0)
            self.model.load_trt(
                '{}/flow.decoder.estimator.{}.sm{}{}.plan'.format(
                    model_dir, 'fp16' if self.fp16 else 'fp32',
                    capability[0], capability[1]
                ),
                '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                trt_concurrent,
                self.fp16
            )

        # HiFi-GAN 缓存字典（用于流式解码）
        self.hift_cache_dict = defaultdict(lambda: None)
```

---

## 模型文件清单

### 1️⃣ **Flow 模型** - 语音 Token 解码器

**文件名**: `flow.pt`

**作用**: 将 LLM 生成的 speech tokens 解码为 mel-spectrogram（声谱特征）

**配置**: `configs['flow']`
- `pre_lookahead_len`: 预测窗口长度（默认 3）
- `token_mel_ratio`: token 与 mel 的比例（2:1）

**推理代码** (cosyvoice/cli/model.py:286-294):
```python
tts_mel, _ = self.flow.inference(
    token=token,                    # LLM 生成的 speech tokens
    token_len=torch.tensor([token.shape[1]]),
    prompt_token=prompt_token,      # 参考音频的 speech tokens
    prompt_token_len=torch.tensor([prompt_token.shape[1]]),
    prompt_feat=prompt_feat,        # 参考音频的特征
    prompt_feat_len=torch.tensor([prompt_feat.shape[1]]),
    embedding=embedding,            # 说话人 embedding
    streaming=stream,               # 是否流式解码
    finalize=finalize               # 是否最后一批
)
```

**输入输出**:
- **输入**: speech tokens (离散单元，来自 LLM)
- **输出**: mel-spectrogram (声谱特征)

---

### 2️⃣ **HiFi-GAN 模型** (hift) - 声码器

**文件名**: `hift.pt`

**作用**: 将 mel-spectrogram 转换为最终音频波形

**配置**: `configs['hift']`

**推理代码** (cosyvoice/cli/model.py:304):
```python
tts_speech, tts_source = self.hift.inference(
    speech_feat=tts_mel,            # Flow 输出的 mel-spectrogram
    cache_source=hift_cache_source   # 缓存源（用于流式拼接）
)
```

**输入输出**:
- **输入**: mel-spectrogram (声谱特征)
- **输出**: 音频波形 (PCM 音频)

---

### 3️⃣ **LLM 模型** - 加载后删除

**文件名**: `llm.pt`

**作用**: Decode 模块加载后立即删除，不使用

**原因**:
- LLM 模块在独立的 `tts_llm` 进程中运行
- Decode 只需要 Flow 和 HiFi-GAN
- 删除 LLM 节省 GPU 内存

**代码** (model_rpc.py:34):
```python
self.model.load(
    '{}/llm.pt'.format(model_dir),
    '{}/flow.pt'.format(model_dir),
    '{}/hift.pt'.format(model_dir)
)
del self.model.llm  # 立即删除，释放内存
```

---

## token2wav 方法详解

**文件**: `cosyvoice/cosyvoice/cli/model.py:284-318`

### 方法签名

```python
def token2wav(
    self,
    token,              # LLM 生成的 speech tokens
    prompt_token,       # 参考音频的 speech tokens
    prompt_feat,        # 参考音频的 mel 特征
    embedding,          # 说话人 embedding
    token_offset,       # token 偏移量（流式解码使用）
    uuid,               # 请求唯一标识
    stream=False,       # 是否流式解码
    finalize=False,     # 是否最后一批
    speed=1.0          # 速度调整（仅非流式）
):
```

### 处理流程

```python
def token2wav(self, token, prompt_token, prompt_feat, embedding, token_offset, uuid, stream=False, finalize=False, speed=1.0):
    with torch.cuda.amp.autocast(self.fp16):
        # 1. Flow 解码：token → mel-spectrogram
        tts_mel, _ = self.flow.inference(
            token=token.to(self.device),
            token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
            prompt_token=prompt_token.to(self.device),
            prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
            prompt_feat=prompt_feat.to(self.device),
            prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
            embedding=embedding.to(self.device),
            streaming=stream,
            finalize=finalize
        )

        # 2. 去除 prompt 部分，只保留生成的 mel
        tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]

        # 3. 拼接缓存（流式解码）
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)

        # 4. HiFi-GAN 解码：mel → audio
        if finalize is False:
            # 流式解码：保留缓存
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)

            # 淡入淡出处理（避免拼接噪声）
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)

            # 更新缓存
            self.hift_cache_dict[uuid] = {
                'mel': tts_mel[:, :, -self.mel_cache_len:],
                'source': tts_source[:, :, -self.source_cache_len:],
                'speech': tts_speech[:, -self.source_cache_len:]
            }

            # 移除重叠部分
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            # 最后一批：不保留缓存
            if speed != 1.0:
                # 速度调整（仅非流式）
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')

            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)

            # 淡入淡出处理
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)

        return tts_speech
```

---

## 核心推理方法：forward()

**文件**: `model_rpc.py:58-85`

```python
@torch.no_grad()
def forward(self, batch: List[DecodeReq]):
    for decode_req in batch:
        # 1. 从共享内存获取数据
        output_ids, speech_index, request_id, token_offset, finalize = decode_req.get_infer_data()
        speech_token, speech_feat, spk_embedding = self.shared_speech_manager.get_index_speech(speech_index)
        speech_token, speech_feat, spk_embedding = speech_token.arr, speech_feat.arr, spk_embedding.arr

        logger.info(f"req_id {request_id} start decode")

        # 2. 调用 token2wav 进行解码
        tts_speech = self.model.token2wav(
            torch.tensor(output_ids, device="cuda").unsqueeze(0),      # LLM 生成的 tokens
            torch.as_tensor(speech_token, device="cuda"),              # 参考 speech tokens
            torch.as_tensor(speech_feat, device="cuda").unsqueeze(0),  # 参考 speech feat
            torch.as_tensor(spk_embedding, device="cuda"),            # 说话人 embedding
            token_offset,                                               # token 偏移
            request_id,                                                # 请求 ID
            stream=decode_req.req.stream,                              # 是否流式
            finalize=finalize,                                        # 是否最后一批
            speed=decode_req.req.speed                                # 速度调整
        )

        # 3. 更新解码状态
        decode_req.update_one_decode(finalize)
        tts_speech = tts_speech.view(-1).cpu().numpy()

        # 4. 返回结果
        if decode_req.req.stream:
            # 流式：推送到队列
            decode_req.req.out_tokens_queue.push(tts_speech, token_offset, finalize)
            logger.info(f"req_id {request_id} decode stream and push")
        else:
            # 非流式：设置完整音频
            logger.info(f"req_id {request_id} decode set_gen_audios")
            decode_req.req.set_gen_audios(tts_speech)

        # 5. 清理缓存
        if finalize:
            self.model.hift_cache_dict.pop(request_id)

    return
```

---

## Manager 类：TTSDecodeManager

**文件**: `light_tts/server/tts_decode/manager.py:28-100`

### 初始化

```python
class TTSDecodeManager:
    def __init__(self, args, tts_decode_port, httpserver_port, style_name, decode_parall_lock, decode_proc_index):
        # ZMQ 通信
        context = zmq.asyncio.Context(2)
        self.recv_from_tts2_gpt = context.socket(zmq.PULL)
        self.recv_from_tts2_gpt.bind(f"{args.zmq_mode}127.0.0.1:{tts_decode_port}")

        self.send_to_httpserver = context.socket(zmq.PUSH)
        self.send_to_httpserver.connect(f"{args.zmq_mode}127.0.0.1:{httpserver_port}")

        # 配置
        configs = load_yaml(args.model_dir)
        self.decode_token_hop_len = 25
        self.flow_pre_lookahead_len = configs["flow"].pre_lookahead_len
        self.speech_token_size = configs["llm"].speech_token_size
        self.eos_id = self.speech_token_size
        self.decode_max_batch_size = 1
```

### 批处理逻辑

```python
def get_batch(self):
    if len(self.waiting_reqs) == 0:
        return []

    batch = []
    appended_reqs = []

    while len(self.waiting_reqs) != 0:
        request_id = self.waiting_reqs.pop(0)
        decode_req = self.req_id_to_out[request_id]

        # 检查输出队列是否已满
        if decode_req.out_queue_is_full():
            appended_reqs.append(request_id)
            continue

        batch.append(decode_req)

        # 批处理大小限制
        if len(batch) >= self.decode_max_batch_size:
            break

    self.waiting_reqs += appended_reqs
    return batch
```

---

## 加速选项

### 1. TensorRT 加速

**文件**:
- `flow.decoder.estimator.fp16.sm{GPU_ARCH}.plan` - TensorRT 引擎
- `flow.decoder.estimator.fp32.onnx` - ONNX 模型

**启用**: `--load_trt True`

**代码** (model_rpc.py:42-47):
```python
if load_trt:
    capability = torch.cuda.get_device_capability(0)  # (8, 6) for RTX 3090
    self.model.load_trt(
        '{}/flow.decoder.estimator.{}.sm{}{}.plan'.format(
            model_dir,
            'fp16' if self.fp16 else 'fp32',
            capability[0], capability[1]
        ),
        '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
        trt_concurrent,
        self.fp16
    )
```

**效果**: Flow decoder 部分（token → mel）使用 TensorRT 加速，推理速度提升 2-3x

---

### 2. JIT 加速

**文件**: `flow.encoder.{fp16/fp32}.zip` - TorchScript 编译的模型

**启用**: `--load_jit True`

**代码** (model_rpc.py:40-41):
```python
if load_jit:
    self.model.load_jit(
        '{}/flow.encoder.{}.zip'.format(
            model_dir,
            'fp16' if self.fp16 else 'fp32'
        )
    )
```

**效果**: Flow encoder 部分使用 TorchScript 加速

---

## ZMQ 通信架构

### Decode 进程的 ZMQ Socket

| Socket | 类型 | 连接 | 用途 |
|--------|------|------|------|
| `recv_from_tts2_gpt` | PULL | bind to `*:tts_decode_port` | 接收 LLM 模块的 tokens |
| `send_to_httpserver` | PUSH | connect to `127.0.0.1:httpserver_port` | 发送音频到 HttpServerManager |

### 数据流向

```
┌─────────────────────┐
│   LLM Process       │
│                     │
│ send_to_tts_decode  │─── ZMQ PUSH ────┐
└─────────────────────┘                  │
                                         │
                                         ↓
┌──────────────────────────────────────────────────┐
│         Decode Process (多个进程)                 │
│                                                  │
│  recv_from_tts2_gpt ← ZMQ PULL (bind)            │
│          ↓                                       │
│  token2wav()                                     │
│     ├─ Flow.inference()  (token → mel)           │
│     └─ hift.inference()   (mel → audio)          │
│          ↓                                       │
│  send_to_httpserver ── ZMQ PUSH ────→            │
└──────────────────────────────────────────────────┘
```

---

## 流式解码详解

### 流式 vs 非流式

**流式解码** (stream=True):
- 分批处理 tokens
- 每批生成部分音频
- 使用缓存实现平滑拼接
- 适合实时应用

**非流式解码** (stream=False):
- 一次性处理所有 tokens
- 生成完整音频
- 支持 speed 调整
- 适合离线应用

### 流式解码流程

```python
# cosyvoice/cli/model.py:338-365
token_offset = 0
while True:
    # 检查是否有足够的 tokens
    this_token_hop_len = self.token_hop_len + prompt_token_pad if token_offset == 0 else self.token_hop_len
    if len(self.tts_speech_token_dict[this_uuid]) - token_offset >= this_token_hop_len + self.flow.pre_lookahead_len:
        # 提取当前批次的 tokens
        this_tts_speech_token = torch.tensor(
            self.tts_speech_token_dict[this_uuid][:token_offset + this_token_hop_len + self.flow.pre_lookahead_len]
        ).unsqueeze(dim=0)

        # 解码当前批次
        this_tts_speech = self.token2wav(
            token=this_tts_speech_token,
            prompt_token=flow_prompt_speech_token,
            prompt_feat=prompt_speech_feat,
            embedding=flow_embedding,
            token_offset=token_offset,
            uuid=this_uuid,
            stream=True,
            finalize=False
        )

        token_offset += this_token_hop_len
        yield {'tts_speech': this_tts_speech.cpu()}

    # 检查是否结束
    if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) - token_offset < this_token_hop_len + self.flow.pre_lookahead_len:
        break

# 处理剩余 tokens（最后一轮）
this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
this_tts_speech = self.token2wav(
    token=this_tts_speech_token,
    prompt_token=flow_prompt_speech_token,
    prompt_feat=prompt_speech_feat,
    embedding=flow_embedding,
    token_offset=token_offset,
    uuid=this_uuid,
    finalize=True  # 标记为最后一轮
)
yield {'tts_speech': this_tts_speech.cpu()}
```

---

## 缓存机制

### HiFi-GAN 缓存

**目的**: 流式解码时，保存部分 mel 和 audio 用于下一轮拼接

**结构** (model.py:307-309):
```python
self.hift_cache_dict[uuid] = {
    'mel': tts_mel[:, :, -self.mel_cache_len:],        # 最后一段 mel
    'source': tts_source[:, :, -self.source_cache_len:], # HiFi-GAN 内部状态
    'speech': tts_speech[:, -self.source_cache_len:]    # 最后一段音频
}
```

**拼接逻辑**:
```python
# 下一轮解码时
if self.hift_cache_dict[uuid] is not None:
    hift_cache_mel = self.hift_cache_dict[uuid]['mel']
    tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)  # 拼接 mel
```

### 淡入淡出处理

**目的**: 避免拼接点产生噪声

**代码** (model.py:306):
```python
tts_speech = fade_in_out(
    tts_speech,                      # 当前生成的音频
    self.hift_cache_dict[uuid]['speech'],  # 缓存的音频
    self.speech_window               # 淡入淡出窗口大小
)
```

---

## 速度调整

**仅支持非流式模式** (model.py:312-314):
```python
if speed != 1.0:
    assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
    # 使用插值调整 mel 长度
    tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
```

**参数**:
- `speed > 1.0`: 加速（压缩 mel）
- `speed < 1.0`: 减速（拉伸 mel）

---

## 性能优化点

### 1. 多进程并行

- **进程数**: `--decode_process_num` 控制
- **GPU 分配**: `gpu_id = decode_proc_index % gpu_num`
- **负载均衡**: HTTP server 通过 round-robin 分发请求

### 2. Batch 处理

```python
self.decode_max_batch_size = 1  # 当前版本 batch size = 1
```

### 3. 半精度推理 (FP16)

```python
with torch.cuda.amp.autocast(self.fp16):
    tts_mel, _ = self.flow.inference(...)
```

### 4. 共享内存

- **读取**: speech_token, speech_feat, spk_embedding
- **零拷贝**: 通过 SharedSpeechManager 避免数据复制

---

## 关键配置参数

### CLI 参数

```bash
--decode_process_num       # Decode 进程数量
--decode_paral_num         # 每个 Decode 进程的并行度
--load_trt                 # 是否加载 TensorRT
--load_jit                 # 是否加载 JIT
```

### 配置文件 (config.yaml)

```yaml
flow:
  pre_lookahead_len: 3     # 预测窗口长度

llm:
  speech_token_size: 4096  # speech token 词汇表大小
```

---

## 实际文件示例

```
pretrained_models/CosyVoice2-0.5B-latest/
├── flow.pt                                      ⭐ Flow 模型权重
├── hift.pt                                      ⭐ HiFi-GAN 模型权重
├── llm.pt                                       # LLM 权重（Decode 加载后删除）
├── config.yaml                                  # 模型配置
│
├── flow.decoder.estimator.fp16.sm86.plan        # TensorRT 引擎（RTX 30xx）
├── flow.decoder.estimator.fp16.sm89.plan        # TensorRT 引擎（RTX 40xx）
├── flow.decoder.estimator.fp32.onnx             # ONNX 模型
└── flow.encoder.fp16.zip                        # TorchScript 模型
```

---

## 与其他模块的对比

| 模块 | 输入 | 输出 | 模型 | 主要功能 |
|------|------|------|------|---------|
| **Encode** | 音频 | tokens, feat, embedding | CosyVoiceFrontEnd | 特征提取 |
| **LLM** | text, prompt | speech tokens | LLM + vLLM | Token 生成 |
| **Decode** | speech tokens | audio waveform | Flow + HiFi-GAN | Token 解码 |

---

## 相关文档

- [SFT_MODE.md](SFT_MODE.md) - SFT 模式使用文档
- [SFT_MODE_DATA_FLOW.md](SFT_MODE_DATA_FLOW.md) - 完整数据流程图
- [ENCODE_MODULE_GUIDE.md](ENCODE_MODULE_GUIDE.md) - Encode 模块详解
- [FRONTEND_CALL_CHAIN.md](FRONTEND_CALL_CHAIN.md) - Frontend 调用链分析

---

## 总结

Decode 模块的核心职责：

1. **接收 LLM 输出**: speech tokens (离散单元)
2. **Flow 解码**: tokens → mel-spectrogram
3. **HiFi-GAN 解码**: mel-spectrogram → 音频波形
4. **流式支持**: 分批处理，缓存拼接
5. **加速优化**: TensorRT, JIT, FP16

**两个关键模型**:
- **Flow**: `flow.pt` - Token → Mel
- **HiFi-GAN**: `hift.pt` - Mel → Audio
