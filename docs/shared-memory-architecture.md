# LightTTS 共享内存架构详解

## 概述

LightTTS 使用跨进程共享内存（Shared Memory）实现高效的音频特征传递,避免了大量数据在不同进程间的序列化/反序列化开销。

## 核心组件

### 1. SharedSpeechManager

**文件**: [light_tts/server/core/objs/shm_speech_manager.py](../light_tts/server/core/objs/shm_speech_manager.py)

**职责**: 管理音频数据在共享内存中的分配、存储和检索

**关键特性**:
- **LRU 缓存**: 相同音频文件只提取一次特征
- **多级存储**: 原始音频 + 提取特征 (token/feat/embedding)
- **跨进程访问**: HTTP Server、Encode、LLM、Decode 模块共享数据
- **spk_id 快速路径**: 预设音色直接通过 `spk_id` 分配,无需 MD5 计算 (⚡ 优化)

**新增方法**:
- `alloc_by_spk_id(spk_id)`: 通过音色 ID 直接分配共享内存,无需 MD5 哈希计算

---

## 数据结构

### 共享内存布局

```python
SharedSpeechManager:
├── preset_slots: int = 10           # 预设音色固定槽位数
├── dynamic_slots: int = 90          # 动态上传槽位数 (size - preset_slots)
├── use_marks: Array[int]           # 使用标记 (0=空闲, 1=已分配, 2=有原始音频, 3=有特征)
├── lru_cache: OrderedDict          # MD5 → speech_index 映射 (仅动态上传)
├── spk_id_to_index: Dict           # spk_id → speech_index 映射 (仅预设音色)
├── lock: ThreadLock                # 并发保护
│
├── prompt_speech_16k_manager       # 原始音频 (16kHz float32)
├── speech_token_manager             # 语音 token (int32)
├── speech_feat_manager              # 语音特征 (float32)
└── spk_embedding_manager            # 说话人嵌入 (float32)

# 槽位分配策略:
# [0, preset_slots)        → 预设音色 (固定槽位, 不参与 LRU)
# [preset_slots, size)     → 动态上传 (LRU 缓存)
```

### 状态机

```
状态 0 (未使用):
  use_marks[index] = 0
  ↓ alloc()

状态 1 (已分配):
  use_marks[index] = 1
  ↓ set_index_data()

状态 2 (有原始音频):
  use_marks[index] = 2
  ↓ set_index_speech()

状态 3 (有完整特征):
  use_marks[index] = 3
  (可被 LLM/Decode 使用)
```

---

## 完整数据流

### 共享内存 speech_index 工作流程

```
┌─────────────────────────────────────────────────────────────┐
│  1. HTTP Server (api_http.py)                              │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  2. 用户上传音频 → load_wav() → alloc_speech_mem()         │
│                                                                 │
│     speech_md5 = calculate_md5(audio)                       │
│     speech_index, have_alloc = alloc_speech_mem(speech_md5) │
│                                                                 │
│     if not have_alloc:                                       │
│         shared_speech_manager.set_index_data(               │
│             speech_index,                                   │
│             shape,                                           │
│             prompt_speech_16k  # 原始音频 (16kHz)            │
│         )                                                     │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Encode Manager (tts_encode/manager.py:102-124)          │
│                                                                 │
│     speech_index = req.speech_index                          │
│                                                                 │
│     if need_extract_speech:                                   │
│         # 从共享内存读取原始音频                               │
│         prompt_speech_16k = shared_speech_manager.get_index_data(speech_index) │
│                                                                 │
│         # 提取特征（关键步骤！）                                │
│         model_input = frontend.frontend_zero_shot(            │
│             '', '', prompt_speech_16k, resample_rate, ''       │
│         )                                                      │
│                                                                 │
│         # 将提取的特征存回共享内存                              │
│         speech_token = model_input["llm_prompt_speech_token"] │
│         speech_feat = model_input["prompt_speech_feat"]       │
│         embedding = model_input["llm_embedding"]              │
│         shared_speech_manager.set_index_speech(               │
│             speech_index, speech_token, speech_feat, embedding │
│         )                                                      │
│     else:                                                     │
│         # 直接从缓存读取已提取的特征                            │
│         speech_token = get_index_speech_token(speech_index)  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  4. LLM & Decode 模块                                        │
│     使用 speech_token 和 embedding 进行推理                   │
└─────────────────────────────────────────────────────────────┘
```

### 数据流详解

#### 阶段 1: HTTP Server 接收请求

**文件**: [api_http.py](../light_tts/server/api_http.py:239-320)

```python
# 1. 用户上传音频
prompt_wav: UploadFile = File(...)
prompt_speech_16k = load_wav(prompt_wav.file, 16000)

# 2. 计算 MD5 (用于缓存查找)
speech_md5 = calculate_md5(prompt_wav.file)

# 3. 分配共享内存索引
speech_index, have_alloc = g_objs.httpserver_manager.alloc_speech_mem(
    speech_md5, prompt_speech_16k
)

# 4. 如果是第一次遇到这个音频,存入原始音频
if not have_alloc:
    g_objs.httpserver_manager.shared_speech_manager.set_index_data(
        speech_index,
        prompt_speech_16k.shape,
        prompt_speech_16k
    )
```

**关键点**:
- `alloc_speech_mem()` 返回 `(index, have_alloc)`
- `have_alloc=True`: LRU 命中缓存,无需重新提取特征
- `have_alloc=False`: 新音频,需要提取特征

---

#### 阶段 2: Encode 模块提取特征

**文件**: [tts_encode/manager.py](../light_tts/server/tts_encode/manager.py:101-124)

```python
# 1. 读取请求
req = shm_req_manager.get_req_obj_by_index(req_index)
speech_index = req.speech_index
need_extract_speech = req.need_extract_speech

# 2. 判断是否需要提取特征
if need_extract_speech:
    # 2a. 从共享内存读取原始音频
    prompt_speech_16k = shared_speech_manager.get_index_data(speech_index)
    prompt_speech_16k = torch.from_numpy(prompt_speech_16k.arr)

    # 2b. 提取特征 (关键步骤!)
    model_input = frontend.frontend_zero_shot(
        '', '', prompt_speech_16k, resample_rate, ''
    )

    # 2c. 将提取的特征存回共享内存
    speech_token = model_input["llm_prompt_speech_token"].cpu().numpy()
    speech_feat = model_input["prompt_speech_feat"].squeeze(0).cpu().numpy()
    embedding = model_input["llm_embedding"].cpu().numpy()

    shared_speech_manager.set_index_speech(
        speech_index, speech_token, speech_feat, embedding
    )
else:
    # 2d. 直接从缓存读取已提取的特征
    speech_token = shared_speech_manager.get_index_speech_token(speech_index).arr[0]

# 3. 传递给 LLM 模块
audio_ids = (speech_token + vocab_size + 2).flatten().tolist()
req.set_speech_token(audio_ids)
```

**特征提取内容**:
| 特征 | 维度 | 用途 |
|------|------|------|
| `speech_token` | [T] | 语音 token 序列 (VQ 编码) |
| `speech_feat` | [T, D] | 语音特征 (用于 Flow 模型) |
| `embedding` | [D] | 说话人嵌入 (用于音色控制) |

---

#### 阶段 3: LLM 模块生成

**文件**: [tts_llm/manager.py](../light_tts/server/tts_llm/manager.py)

```python
# 1. 从共享内存读取请求
req = shm_req_manager.get_req_obj_by_index(req_index)

# 2. 读取语音 token (已由 Encode 模块写入)
audio_ids = req.audio_ids  # 来自 shared_speech_manager

# 3. 组合输入序列
input_ids = [sos_eos] + prompt_text_ids + text_ids + [task_id] + audio_ids

# 4. 调用 LLM 模型生成
output_tokens = model.generate(input_ids, ...)

# 5. 将结果存入共享内存 (供 Decode 读取)
req.out_tokens_queue.push(output_tokens)
```

---

#### 阶段 4: Decode 模块生成音频

**文件**: [tts_decode/manager.py](../light_tts/server/tts_decode/manager.py)

```python
# 1. 从共享内存读取 LLM 输出
output_tokens = req.out_tokens_queue.pop()

# 2. 调用 Flow 模型生成音频
tts_speech = flow_model.decode(output_tokens, ...)

# 3. 写入共享内存 (供 HTTP Server 读取)
req.set_gen_audios(tts_speech)

# 4. 通知 HTTP Server
httpserver_manager.push_result(req_index)
```

---

## LRU 缓存机制

### 核心逻辑

**文件**: [shm_speech_manager.py:91-111](../light_tts/server/core/objs/shm_speech_manager.py:91-111)

```python
def alloc(self, speech_md5):
    with self.lock:
        # 1. 检查缓存
        if speech_md5 in self.lru_cache:
            self.lru_cache.move_to_end(speech_md5)  # 更新访问时间
            return self.lru_cache[speech_md5], True  # 命中缓存

        # 2. 未命中,分配新索引
        if len(self.lru_cache) >= self.size:
            # 2a. 缓存已满,驱逐最久未使用的项
            key, value = self.lru_cache.popitem(last=False)
            index = value
        else:
            # 2b. 找到空闲槽位
            for i in range(self.size):
                if self.use_marks.arr[i] == 0:
                    index = i
                    break

        # 3. 标记为已分配
        self.use_marks.arr[index] = 1
        self.lru_cache[speech_md5] = index
        return index, False  # 新分配
```

### 性能优势

| 场景 | 无缓存 | 有 LRU 缓存 |
|------|--------|-----------|
| **首次请求** | 提取特征 (~300ms) | 提取特征 (~300ms) |
| **重复请求** | 提取特征 (~300ms) | 命中缓存 (~0ms) ⚡ |
| **内存占用** | - | 可配置 (默认 100 项) |

**示例**:
```python
# 请求 1: 上传 audio.wav
speech_index, have_alloc = alloc_speech_mem(md5, audio)
# have_alloc = False (需要提取特征)

# 请求 2-100: 上传相同的 audio.wav
speech_index, have_alloc = alloc_speech_mem(md5, audio)
# have_alloc = True (直接使用缓存,跳过特征提取)
```

---

## 预设音色优化 (Preset Voices Optimization)

### 优化目标

消除预设音色请求中的 MD5 计算开销,通过 `spk_id` 直接映射到共享内存索引。

### 架构改进

#### 优化前流程

```
用户请求 spk_id="male"
  ↓
加载音频文件 (assets/male.mp3)
  ↓
计算 MD5 (耗时 ~5-10ms)
  ↓
alloc_speech_mem(md5, audio) → LRU 查找
  ↓
返回 speech_index
```

#### 优化后流程 ⚡

```
用户请求 spk_id="male"
  ↓
alloc_speech_mem(spk_id="male") → 直接映射
  ↓
返回 speech_index (预加载时已分配)
```

### 关键实现

#### 1. SharedSpeechManager 固定槽位设计

**文件**: [shm_speech_manager.py](../light_tts/server/core/objs/shm_speech_manager.py:75-188)

**核心改进**: 使用固定槽位隔离预设音色和动态上传

```python
class SharedSpeechManager:
    def __init__(self, name, size, init_mark=True, preset_slots=10):
        self.preset_slots = preset_slots           # 预设音色槽位数 (默认 10)
        self.dynamic_slots = size - preset_slots   # 动态上传槽位数
        self.spk_id_to_index = {}                  # spk_id → 固定槽位映射

    def alloc_by_spk_id(self, spk_id: str) -> Tuple[int, bool]:
        """预设音色: 使用固定槽位 [0, preset_slots), 不参与 LRU"""
        with self.lock:
            if spk_id in self.spk_id_to_index:
                return self.spk_id_to_index[spk_id], True  # 命中缓存

            # 在预设槽位范围内分配
            if len(self.spk_id_to_index) >= self.preset_slots:
                raise RuntimeError(f"preset slots full ({self.preset_slots})")

            index = self._find_free_slot_in_range(0, self.preset_slots)
            self.spk_id_to_index[spk_id] = index
            return index, False  # 新分配

    def alloc(self, speech_md5: str) -> Tuple[int, bool]:
        """动态上传: 使用 LRU 槽位 [preset_slots, size)"""
        with self.lock:
            if speech_md5 in self.lru_cache:
                return self.lru_cache[speech_md5], True

            # 在动态槽位范围内分配 (支持 LRU 驱逐)
            index = self._find_or_evict_in_range(self.preset_slots, self.size)
            self.lru_cache[speech_md5] = index
            return index, False
```

**槽位分配示意图**:
```
索引:  0   1   2   ...   9  |  10  11  12  ...  99
       ├──┬──┬──┬──┬──┤   ├─┬──┬──┬──┬──┤
       │  │  │  │  │  │   │  │  │  │  │
       └──┴──┴──┴──┴──┘   └──┴──┴──┴──┴──┘
       预设音色 (固定)      动态上传 (LRU)
       [0, 10)             [10, 100)
```

#### 2. SpeakerManager 预加载共享内存

**文件**: [speaker_manager.py](../light_tts/server/speaker_manager.py:128-165)

**优化**: 预计算 `semantic_len` 并存储

```python
class SpeakerManager:
    def __init__(self, model, yaml_path, shared_speech_manager=None):
        self.shared_speech_manager = shared_speech_manager

    def _register_voice(self, spk_id, audio_path, prompt_text):
        # 提取特征
        model_input = self.model.frontend_zero_shot(...)

        # 预分配共享内存 (固定槽位)
        speech_index, have_alloc = self.shared_speech_manager.alloc_by_spk_id(spk_id)

        # 预计算语义长度
        semantic_len = (prompt_speech_16k.shape[1] + 239) // 640 + 10

        if not have_alloc:
            # 存储原始音频
            self.shared_speech_manager.set_index_data(
                speech_index, prompt_speech_16k.shape, prompt_speech_16k
            )

            # 存储特征
            self.shared_speech_manager.set_index_speech(
                speech_index, speech_token, speech_feat, embedding
            )

        # 存储完整信息 (包括 semantic_len)
        self.voices[spk_id] = {
            'audio_path': audio_path,
            'prompt_text': prompt_text,
            'speech_index': speech_index,
            'semantic_len': semantic_len,  # 预计算值
        }
```

#### 3. HTTP API 快速路径

**文件**: [api_http.py](../light_tts/server/api_http.py:287-302)

```python
@app.post("/inference_zero_shot")
async def inference_zero_shot(spk_id: str = Form(default=""), ...):
    if spk_id:
        # ========== 预设音色快速路径 ==========
        speech_index, have_alloc = g_objs.httpserver_manager.alloc_speech_mem(spk_id=spk_id)

        # 无需 MD5 计算,无需加载音频文件
        voice_info = g_objs.speaker_manager.get_voice_info(spk_id)
        prompt_text = voice_info['prompt_text']
        semantic_len = voice_info.get('semantic_len', 0)  # 使用预计算值
        need_extract_speech = False  # 特征已在启动时提取
    else:
        # ========== 动态上传路径 ==========
        prompt_speech_16k = load_wav(prompt_wav.file, 16000)
        semantic_len = (prompt_speech_16k.shape[1] + 239) // 640 + 10
        speech_md5 = calculate_md5(prompt_wav.file)
        speech_index, have_alloc = g_objs.httpserver_manager.alloc_speech_mem(
            speech_md5=speech_md5, prompt_wav=prompt_speech_16k
        )
        need_extract_speech = not have_alloc
```

### 性能对比

| 操作 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| **MD5 计算** | ~5-10ms | 0ms | ✅ 100% |
| **音频加载** | 每次请求 | 启动时一次 | ✅ 减少 I/O |
| **特征提取** | 每次请求 | 启动时一次 | ✅ 减少 CPU |
| **内存查找** | MD5 Hash Map | spk_id Dict | ✅ 更快 |
| **LRU 驱逐风险** | 可能被驱逐 | **固定槽位,永不驱逐** | ✅ 100% |

### 启动时预加载流程

```
服务启动
  ↓
SpeakerManager.load_presets()
  ↓
遍历 voices.yaml
  ↓
对每个音色:
  1. 加载音频文件 (启动时一次)
  2. 提取特征 (启动时一次)
  3. alloc_by_spk_id(spk_id) → 分配固定槽位 [0, preset_slots)
  4. 存储原始音频 + 特征到共享内存
  5. 预计算 semantic_len 并存储到 voices 字典
  ↓
预设音色就绪,固定槽位,永不驱逐
```

### 请求处理流程

```
POST /inference_zero_shot?spk_id=male&tts_text=你好
  ↓
alloc_speech_mem(spk_id="male")
  ↓
查询 spk_id_to_index["male"] → speech_index=2 (固定槽位)
  ↓
Encode 模块从 speech_index=2 读取已缓存的特征
  ↓
LLM/Decode 模块生成音频
```

### 固定槽位优势

1. **零驱逐风险**: 预设音色使用固定槽位,永远不会被 LRU 驱逐
2. **性能可预测**: 预设音色响应时间恒定,不受缓存状态影响
3. **资源隔离**: 预设音色和动态上传完全隔离,互不影响
4. **易于监控**: 可精确统计预设音色和动态上传的资源使用

### 配置建议

| 场景 | preset_slots | cache_capacity | 说明 |
|------|-------------|----------------|------|
| **小规模** | 5 | 50 | 5 个预设音色,45 个动态上传 |
| **中规模** | 10 | 100 | 10 个预设音色,90 个动态上传 |
| **大规模** | 20 | 200 | 20 个预设音色,180 个动态上传 |

**计算公式**:
```
cache_capacity = preset_slots + 动态上传槽位数
dynamic_slots = cache_capacity - preset_slots
```

---

## 并发控制

### 锁机制

```python
class SharedSpeechManager:
    def __init__(self):
        self.lock = threading.Lock()  # 进程内线程锁

    def alloc(self, speech_md5):
        with self.lock:  # 保护 LRU 缓存操作
            ...

    def set_index_data(self, index, shape, data):
        # 无锁 (写入独立共享内存区域)
        ...
```

### 跨进程同步

```python
# 使用 SharedMemory (multiprocessing.shared_memory)
# - Python 3.8+ 原生支持
# - 自动映射到同一块物理内存
# - 进程间自动同步

class SharedArray:
    def __init__(self, name, shape, dtype):
        self.shm = shared_memory.SharedMemory(name=name, create=True, size=dest_size)
        self.arr = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)
```

---

## 关键常量

### 参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cache_capacity` | 100 | LRU 缓存大小 |
| `resample_rate` | 24000 | 目标采样率 |
| `token_hop_len` | 25 | Token 帧移 |

### 计算公式

```python
# 语义长度计算
semantic_len = (prompt_speech_16k.shape[1] + 239) // 640 + 10

# Token 填充
prompt_token_pad = int(np.ceil(speech_token.size / token_hop_len) * token_hop_len - speech_token.size)

# 音频 ID 偏移 (避免与文本 token 冲突)
audio_ids = (speech_token + vocab_size + 2)
```

---

## 性能优化要点

### 1. 零拷贝传输

```python
# ❌ 错误做法 (序列化开销)
pickle_data = pickle.dumps(audio_tensor)
send_to_another_process(pickle_data)

# ✅ 正确做法 (共享内存)
shm_arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
shm_arr[:] = audio_tensor  # 直接内存写入,无序列化
```

### 2. 懒加载特征

```python
if need_extract_speech:
    # 只有第一次才提取特征
    extract_features()
else:
    # 后续请求直接使用缓存
    use_cached_features()
```

### 3. LRU 驱逐策略

```python
# 驱逐最久未使用的项
key, value = self.lru_cache.popitem(last=False)

# 释放共享内存
self.use_marks.arr[value] = 0
```

---

## 故障排查

### 常见问题

#### 1. `speech_index = None`

**错误**:
```python
TypeError: 'NoneType' object cannot be interpreted as an integer
```

**原因**:
- 未调用 `alloc_speech_mem()` 分配索引
- 或使用了预设音色但未实现共享内存逻辑

**解决**:
```python
# 确保分配索引
speech_index, have_alloc = alloc_speech_mem(speech_md5, prompt_speech_16k)

# 或使用预设音色时,也要加载音频
if spk_id:
    prompt_speech_16k = load_wav(audio_path, 16000)
    speech_index, _ = alloc_speech_mem(speech_md5, prompt_speech_16k)
```

#### 2. 缓存未命中

**现象**: 每次请求都重新提取特征

**检查**:
```python
# 确认 MD5 计算正确
speech_md5 = calculate_md5(file_object)

# 确认缓存未满
print(f"LRU cache size: {len(lru_cache)} / {cache_capacity}")
```

#### 3. 内存泄漏

**症状**: `use_marks` 全部为 1, 无法分配新索引

**解决**:
```python
# 重启服务,或增加 cache_capacity
--cache_capacity 200  # 默认 100
```

---

## 监控指标

### 关键指标

```python
# 缓存命中率
cache_hit_rate = hit_count / (hit_count + miss_count)

# 共享内存使用率
usage_rate = len(lru_cache) / cache_capacity

# 平均特征提取时间
avg_extract_time = total_extract_time / extract_count
```

### 日志输出

```python
# Encode 模块日志
logger.info(f"req_id {req.request_id} generate speech index {speech_index} cache")
logger.info(f"req_id {req.request_id} use speech index {speech_index} cache")
```

---

## 总结

### 架构优势

1. **高性能**: 零拷贝跨进程传输,避免序列化开销
2. **智能缓存**: LRU 自动复用已提取的特征
3. **可扩展**: 支持多进程并发,易于水平扩展
4. **低延迟**: 共享内存比网络传输快 10-100 倍

### 设计原则

1. **特征提取集中化**: Encode 模块统一提取
2. **数据共享最大化**: 多模块复用同一份数据
3. **内存管理自动化**: LRU 自动驱逐,无需手动释放
4. **并发安全**: 线程锁 + 进程隔离

---

## 参考资料

- [multiprocessing.shared_memory](https://docs.python.org/3/library/multiprocessing.shared_memory.html)
- [LRU Cache 实现](https://docs.python.org/3/library/collections.html#collections.OrderedDict)
- [ZMQ 模式](https://zeromq.org/get-started/?language=python&framework=asyncio)
