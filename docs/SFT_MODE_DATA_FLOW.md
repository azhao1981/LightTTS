# SFT 模式数据流程图

## 整体架构

LightTTS 采用 **Encode-LLM-Decode 三阶段流水线架构**，每个阶段作为独立进程运行：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         HTTP Client Request                             │
│              POST /inference_zero_shot                                  │
│              { spk_id: "female_test_sft", tts_text: "你好" }            │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      1️⃣ API Layer (api_http.py)                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  📍 修复点 1: 提取 prompt_text_ids (第 325-334 行)                │  │
│  │                                                                   │  │
│  │  1. 检测 SFT 模式: is_sft = spk_id.endswith('_sft')               │  │
│  │  2. 获取 base_spk_id = "female_test"                             │  │
│  │  3. 从 spk2info.pt 提取数据:                                      │  │
│  │     - llm_embedding → 说话人 embedding                            │  │
│  │     - llm_prompt_speech_token → 0-based speech tokens             │  │
│  │     - prompt_text → 参考文本 token IDs ⭐ 新增                   │  │
│  │     - prompt_speech_feat → 音频特征                               │  │
│  │  4. 存储到共享内存 (SharedSpeechManager)                          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  ↓                                                                      │
│  构建 request_dict = {                                                  │
│    spk_id: "female_test_sft",                                           │
│    prompt_text_ids: [123, 456, ...], ⭐ 新增                           │
│    speech_index: 42,                                                    │
│    semantic_len: 89,                                                    │
│    need_extract_speech: False,                                          │
│    ...                                                                  │
│  }                                                                      │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────────┐
│              2️⃣ HttpServerManager (httpserver/manager.py)              │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  📍 修复点 2: 使用预计算的 prompt_text_ids (第 201-227 行)        │  │
│  │                                                                   │  │
│  │  1. 检查 prompt_text_ids 字段                                     │  │
│  │  2. 如果有: 直接使用 (SFT 模式) ⭐ 新增                           │  │
│  │  3. 如果无: 实时编码 prompt_text (Zero-Shot 模式)                 │  │
│  │                                                                   │  │
│  │  prompt_ids = [sos_eos] + prompt_text_ids + text_ids + [task_id] │  │
│  │     ↓              ↓                ↓            ↓              ↓  │  │
│  │  [vocab_size]   [0, vocab)    [0, vocab)  [vocab+1]            │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  ↓                                                                      │
│  初始化 Req 对象，分配共享内存索引                                      │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────────┐
│               3️⃣ Encode Manager (tts_encode/manager.py)                │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  📍 修复点 3: 添加 vocab offset (第 131-143 行)                   │  │
│  │                                                                   │  │
│  │  if is_sft:                                                       │  │
│  │    1. 从共享内存读取:                                             │  │
│  │       (speech_token, speech_feat, embedding)                      │  │
│  │                                                                   │  │
│  │    2. ⭐ 关键修复: 添加 vocab offset                             │  │
│  │       speech_token_offset = speech_token + vocab_size + 2        │  │
│  │                                                                   │  │
│  │    3. 计算 prompt_token_pad                                       │  │
│  │                                                                   │  │
│  │    4. 发送到 LLM                                                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  ↓                                                                      │
│  ZMQ PUSH → Send req.index_in_shm_mem to tts_llm                       │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    4️⃣ LLM Module (tts_llm/)                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  接收完整 token 序列:                                              │  │
│  │                                                                   │  │
│  │  [sos_eos] + [prompt_text_ids] + [text_ids] + [task_id]          │  │
│  │  + [speech_tokens_with_offset]                                    │  │
│  │                                                                   │  │
│  │  1. 查询 LLM Embedding Table:                                     │  │
│  │     - [0, vocab_size) → text embeddings                          │  │
│  │     - [vocab_size] → sos_eos embedding                           │  │
│  │     - [vocab_size+1] → task_id embedding                         │  │
│  │     - [vocab_size+2, ...) → speech embeddings ⭐ 正确位置        │  │
│  │                                                                   │  │
│  │  2. 生成目标语音 token 序列                                        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  ↓                                                                      │
│  ZMQ PUSH → Send speech tokens to tts_decode                           │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                  5️⃣ Decode Module (tts_decode/)                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  模型文件:                                                        │  │
│  │  - flow.pt   (Token → Mel-spectrogram)                           │  │
│  │  - hift.pt   (Mel → Audio waveform, HiFi-GAN vocoder)           │  │
│  │  - llm.pt    (加载后删除，不使用)                                │  │
│  │                                                                  │  │
│  │  加速选项:                                                        │  │
│  │  - TensorRT: flow.decoder.estimator.fp16.sm*.plan               │  │
│  │  - JIT:      flow.encoder.fp16.zip                              │  │
│  │                                                                  │  │
│  │  处理流程:                                                        │  │
│  │  1. 接收 LLM 生成的 speech tokens                                │  │
│  │  2. 从共享内存读取: speech_token, speech_feat, spk_embedding     │  │
│  │  3. Flow.inference(): token → mel-spectrogram                   │  │
│  │  4. HiFi-GAN.inference(): mel → audio waveform                 │  │
│  │  5. 支持流式解码（缓存拼接）或非流式（完整生成）                 │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  ↓                                                                      │
│  ZMQ PUSH → Send audio data to HttpServerManager                       │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    6️⃣ HTTP Response                                    │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  返回音频流或完整音频文件:                                         │  │
│  │  - Streaming: 分块返回音频数据                                    │  │
│  │  - Non-streaming: 一次性返回完整 WAV                              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 三个修复点的详细说明

### 📍 修复点 1: API Layer - 提取 prompt_text_ids

**文件**: `light_tts/server/api_http.py` (第 325-334 行)

**问题**: SFT 模式缺少 prompt_text，导致 LLM 输入结构不完整

**修复**:
```python
# 从 spk2info 提取 prompt_text token IDs
prompt_text_tensor = spk_info.get('prompt_text', torch.tensor([]))
if prompt_text_tensor.numel() > 0:
    prompt_text_ids = prompt_text_tensor.flatten().tolist()
else:
    prompt_text_ids = []
```

**数据流**:
```
spk2info.pt
  └─['prompt_text'] (tensor [1, 89])
        ↓
      prompt_text_ids (list [89 tokens])
        ↓
      request_dict['prompt_text_ids']
        ↓
      HttpServerManager
```

---

### 📍 修复点 2: HttpServerManager - 使用预计算的 prompt_text_ids

**文件**: `light_tts/server/httpserver/manager.py` (第 201-227 行)

**问题**: 没有使用预计算的 prompt_text_ids，仍然尝试实时编码

**修复**:
```python
# 使用预计算的 prompt_text_ids（如果提供）
prompt_text_ids = request_dict.get("prompt_text_ids")
if prompt_text_ids is not None:
    logger.info(f"using pre-computed prompt_text_ids, len={len(prompt_text_ids)}")
else:
    prompt_text_ids = await self._async_encode(request_dict["prompt_text"])

# 构建完整的 prompt_ids
prompt_ids = list(chain([self.sos_eos], prompt_text_ids, text_ids, [self.task_id]))
```

**数据流**:
```
request_dict
  └─['prompt_text_ids'] (list [89 tokens])
        ↓
      prompt_ids = [sos_eos, prompt_text_ids, text_ids, task_id]
        ↓
      Req object (共享内存)
        ↓
      Encode Manager
```

---

### 📍 修复点 3: Encode Manager - 添加 vocab offset

**文件**: `light_tts/server/tts_encode/manager.py` (第 131-143 行)

**问题**: speech_token 没有 vocab offset，导致查表错误

**修复**:
```python
# 添加 vocab_size + 2 offset
if speech_token.size > 0:
    speech_token_offset = (speech_token + self.vocab_size + 2)
    audio_ids = speech_token_offset.flatten().tolist()
else:
    audio_ids = []
```

**数据流**:
```
spk2info.pt
  └─['llm_prompt_speech_token'] (0-based [0, 382))
        ↓
      speech_token_offset = speech_token + vocab_size + 2
        ↓
      audio_ids (mapped to [vocab_size+2, ...))
        ↓
      req.speech_token
        ↓
      LLM Embedding Table (正确查询 speech tokens 区域)
```

---

## 修复前后对比

### 修复前 (v1.0.0)

```
API Layer
  └─ prompt_text = "" ❌
  └─ prompt_text_ids = undefined ❌

HttpServerManager
  └─ 尝试编码空字符串 ❌
  └─ prompt_ids = [sos_eos] + [] + [text_ids] + [task_id] ❌

Encode Manager
  └─ audio_ids = speech_token (0-based) ❌
  └─ 映射到 text tokens 区域 ❌

LLM
  └─ 查表错误 → 生成乱码 ❌
```

### 修复后 (v1.0.1)

```
API Layer
  └─ prompt_text_ids = [89 tokens] ✅ (从 spk2info 提取)

HttpServerManager
  └─ prompt_ids = [sos_eos] + [89 tokens] + [text_ids] + [task_id] ✅

Encode Manager
  └─ audio_ids = speech_token + vocab_size + 2 ✅
  └─ 映射到 speech tokens 区域 ✅

LLM
  └─ 查表正确 → 生成正常音频 ✅
```

---

## 关键数据结构

### spk2info.pt 结构

```python
spk2info = {
    'female_test': {
        'llm_embedding': Tensor([1, 192]),           # 说话人 embedding
        'llm_prompt_speech_token': Tensor([382]),    # 0-based speech tokens
        'prompt_text': Tensor([1, 89]),              # 参考文本 token IDs ⭐
        'prompt_speech_feat': Tensor([t, 80]),       # 音频特征
    },
    'male1_trained': { ... }
}
```

### LLM Embedding Table 索引

```
索引范围              │ 内容            │ 来源
─────────────────────┼─────────────────┼──────────────────
[0, vocab_size)      │ text tokens     │ prompt_text, text
[vocab_size]         │ sos_eos         │ 固定值
[vocab_size+1]       │ task_id         │ 固定值
[vocab_size+2, ...)  │ speech tokens   │ llm_prompt_speech_token + offset ⭐
```

---

## 通信机制

### 进程间通信 (IPC)

1. **HTTP → HttpServerManager**: 函数调用（同进程）
2. **HttpServerManager → Encode**: ZMQ PUSH
3. **Encode → LLM**: ZMQ PUSH
4. **LLM → Decode**: ZMQ PUSH
5. **Decode → HttpServerManager**: ZMQ PUSH
6. **HttpServerManager → HTTP**: 函数调用（同进程）

### 共享内存

- **ShmReqManager**: 存储 Req 对象（包含 prompt_ids, speech_token）
- **SharedSpeechManager**: 存储 speech_token, speech_feat, embedding

---

## 性能优化点

### SFT vs Zero-Shot

| 项目 | Zero-Shot | SFT | 优化 |
|------|-----------|-----|------|
| prompt_text 编码 | 实时编码 | 预计算（spk2info） | 跳过前端编码 |
| embedding 提取 | 实时提取 | 预计算（spk2info） | 跳过模型推理 |
| speech_token 编码 | 实时提取 | 预计算（spk2info） | 跳过 Codec 编码 |
| 数据传输量 | 12 字段 | 4 字段 | -67% |

---

## 相关文档

- **完整模块指南**：
  - [ENCODE_MODULE_GUIDE.md](ENCODE_MODULE_GUIDE.md) - Encode 模块详解
  - [DECODE_MODULE_GUIDE.md](DECODE_MODULE_GUIDE.md) - Decode 模块详解 ⭐
- [SFT_MODE.md](SFT_MODE.md) - SFT 模式使用文档
- [sft_bug.md](sft_bug.md) - Bug 分析与修复记录
- [FRONTEND_CALL_CHAIN.md](FRONTEND_CALL_CHAIN.md) - Frontend 调用链分析
- [../README.md](../README.md) - 项目整体说明
