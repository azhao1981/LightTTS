# LightTTS Frontend 调用链分析

## 核心函数定义位置

### CosyVoice Frontend 类

**文件**: `cosyvoice/cosyvoice/cli/frontend.py`

```python
class CosyVoiceFrontEnd:
    def frontend_sft(self, tts_text, spk_id):
        """SFT 模式：使用预训练说话人"""
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        embedding = self.spk2info[spk_id]['embedding']
        model_input = {
            'text': tts_text_token,
            'text_len': tts_text_token_len,
            'llm_embedding': embedding,
            'flow_embedding': embedding
        }
        return model_input

    def frontend_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, resample_rate, zero_shot_spk_id):
        """Zero-Shot 模式：使用参考音频"""
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)

        if zero_shot_spk_id == '':
            # 实时提取特征
            prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)
            prompt_speech_resample = torchaudio.transforms.Resample(16000, resample_rate)(prompt_speech_16k)
            speech_feat, speech_feat_len = self._extract_speech_feat(prompt_speech_resample)
            speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)
            embedding = self._extract_spk_embedding(prompt_speech_16k)

            model_input = {
                'prompt_text': prompt_text_token,
                'prompt_text_len': prompt_text_token_len,
                'llm_prompt_speech_token': speech_token,
                'llm_prompt_speech_token_len': speech_token_len,
                'flow_prompt_speech_token': speech_token,
                'flow_prompt_speech_token_len': speech_token_len,
                'prompt_speech_feat': speech_feat,
                'prompt_speech_feat_len': speech_feat_len,
                'llm_embedding': embedding,
                'flow_embedding': embedding
            }
        else:
            # 使用缓存的 spk2info 数据
            model_input = self.spk2info[zero_shot_spk_id]

        model_input['text'] = tts_text_token
        model_input['text_len'] = tts_text_token_len
        return model_input
```

---

## 完整调用链

### 1️⃣ SFT 模式调用链

```
HTTP Request
  { spk_id: "female_test_sft", tts_text: "你好" }
        ↓
┌─────────────────────────────────────────────────────┐
│ api_http.py (第 389-448 行)                         │
│                                                     │
│ 1. 检测 SFT 模式: is_sft = spk_id.endswith('_sft') │
│ 2. base_spk_id = "female_test"                     │
│ 3. 直接从 spk2info 读取（不调用 frontend_sft）      │
│                                                     │
│    spk_info = g_objs.frontend.spk2info[base_spk_id]│
│    llm_embedding = spk_info['llm_embedding']        │
│    speech_token = spk_info['llm_prompt_speech_token']│
│    prompt_text_ids = spk_info['prompt_text']       │
│                                                     │
│ 4. 存储到共享内存                                   │
└─────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────┐
│ httpserver/manager.py (第 201-227 行)               │
│                                                     │
│  使用预计算的 prompt_text_ids 构建序列              │
│  prompt_ids = [sos_eos] + prompt_text_ids + ...     │
└─────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────┐
│ tts_encode/manager.py (第 110-146 行)               │
│                                                     │
│  SFT 分支:                                          │
│  - 从共享内存读取 embedding                         │
│  - 添加 vocab offset ⭐ v1.0.1 修复                 │
│  - 发送到 LLM                                       │
└─────────────────────────────────────────────────────┘
```

**关键点**: SFT 模式 **不调用** `frontend_sft()`，而是直接从 `spk2info` 读取预计算的数据。

---

### 2️⃣ Zero-Shot 模式调用链（预设音色）

```
HTTP Request
  { spk_id: "female", tts_text: "你好" }
        ↓
┌─────────────────────────────────────────────────────┐
│ api_http.py (第 450-464 行)                         │
│                                                     │
│  Zero-shot 模式:                                    │
│  - 验证 spk_id 有效性                               │
│  - 从 voices.yaml 获取 prompt_text                  │
│  - 分配共享内存                                     │
└─────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────┐
│ tts_encode/manager.py (第 150-179 行)               │
│                                                     │
│  if need_extract_speech:                            │
│    prompt_speech_16k = shared_speech_manager.get()  │
│    model_input = self.frontend.frontend_zero_shot( │
│        '', '', prompt_speech_16k, 24000, ''         │
│    ) ⭐ 调用 frontend_zero_shot                     │
│                                                     │
│  - 提取 speech_token, speech_feat, embedding        │
│  - 添加 vocab offset                               │
│  - 发送到 LLM                                       │
└─────────────────────────────────────────────────────┘
```

---

### 3️⃣ Zero-Shot 模式调用链（动态上传音频）

```
HTTP Request
  { spk_id: "", prompt_wav: <audio>, tts_text: "你好" }
        ↓
┌─────────────────────────────────────────────────────┐
│ api_http.py (第 317-464 行)                         │
│                                                     │
│  1. 接收上传的音频文件                              │
│  2. 计算音频 MD5                                    │
│  3. 分配共享内存 (基于 MD5)                         │
│  4. 存储音频到共享内存                              │
└─────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────┐
│ speaker_manager.py (第 116-149 行)                  │
│                                                     │
│  预提取特征（首次使用时）:                           │
│  prompt_speech_16k = load_wav(audio_path)          │
│  model_input = self.model.frontend_zero_shot(      │
│      '', prompt_text, prompt_speech_16k, 24000, ''  │
│  ) ⭐ 调用 frontend_zero_shot                       │
│                                                     │
│  存储到:                                            │
│  - self.model.spk2info[spk_id] = model_input       │
│  - shared_speech_manager.set_index_speech(...)     │
└─────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────┐
│ tts_encode/manager.py (第 150-179 行)               │
│                                                     │
│  使用缓存的特征（不调用 frontend）                  │
│  - 从共享内存读取 speech_token, speech_feat         │
│  - 添加 vocab offset                               │
│  - 发送到 LLM                                       │
└─────────────────────────────────────────────────────┘
```

---

## 调用位置总结

| 调用场景 | 调用位置 | 函数 | 是否调用 frontend |
|---------|---------|------|------------------|
| **SFT 模式** | api_http.py | ❌ 不调用 | 直接读取 spk2info |
| **Zero-Shot (预设)** | tts_encode/manager.py | ✅ frontend_zero_shot | 实时提取特征 |
| **Zero-Shot (动态)** | speaker_manager.py | ✅ frontend_zero_shot | 预提取并缓存 |
| **Zero-Shot (缓存后)** | tts_encode/manager.py | ❌ 不调用 | 使用缓存特征 |

---

## 为什么 SFT 模式不调用 frontend_sft？

### 原因分析

1. **spk2info 已包含所有数据**
   ```python
   # spk2info['female_test'] 包含:
   {
       'llm_embedding': Tensor([1, 192]),
       'llm_prompt_speech_token': Tensor([382]),
       'prompt_text': Tensor([1, 89]),
       'prompt_speech_feat': Tensor([t, 80]),
   }
   ```

2. **frontend_sft 返回的数据不完整**
   ```python
   # cosyvoice/cli/frontend.py: frontend_sft
   model_input = {
       'text': tts_text_token,
       'llm_embedding': embedding,
       'flow_embedding': embedding
   }
   # ❌ 缺少 prompt_text, speech_token, speech_feat
   ```

3. **SFT 需要完整的 zero-shot 格式数据**
   - SFT 模式复用 zero-shot 的数据结构
   - 需要 prompt_text (89 tokens) 用于 LLM 输入
   - 需要 speech_token (需要加 vocab offset)

### v1.0.1 修复的关键

```python
# api_http.py (第 410-419 行) ⭐ 新增
prompt_text_tensor = spk_info.get('prompt_text', torch.tensor([]))
if prompt_text_tensor.numel() > 0:
    prompt_text_ids = prompt_text_tensor.flatten().tolist()
else:
    prompt_text_ids = []

# tts_encode/manager.py (第 133-136 行) ⭐ 修复
speech_token_offset = (speech_token + self.vocab_size + 2)
audio_ids = speech_token_offset.flatten().tolist()
```

---

## 数据流对比

### frontend_zero_shot 输出

```python
{
    'prompt_text': Tensor([1, 89]),              # 参考文本 tokens
    'llm_prompt_speech_token': Tensor([382]),    # 0-based speech tokens
    'prompt_speech_feat': Tensor([t, 80]),       # 音频特征
    'llm_embedding': Tensor([1, 192]),           # 说话人 embedding
    'text': Tensor([1, text_len]),               # 用户文本 tokens
}
```

### spk2info 结构 (SFT/Zero-Shot 缓存)

```python
{
    'prompt_text': Tensor([1, 89]),              # ⭐ v1.0.1 新增使用
    'llm_prompt_speech_token': Tensor([382]),    # 需要 + vocab_size + 2
    'prompt_speech_feat': Tensor([t, 80]),
    'llm_embedding': Tensor([1, 192]),
}
```

---

## 关键代码位置索引

### 函数定义
- **frontend_sft**: `cosyvoice/cosyvoice/cli/frontend.py:151`
- **frontend_zero_shot**: `cosyvoice/cosyvoice/cli/frontend.py:157`

### 调用位置
1. **speaker_manager.py**: `light_tts/server/speaker_manager.py:121`
   - 预设音色的特征提取和缓存

2. **tts_encode/manager.py**: `light_tts/server/tts_encode/manager.py:158`
   - Zero-Shot 模式实时提取特征

3. **api_http.py**: SFT 模式不调用 frontend，直接读取 spk2info

### 修复点
1. **api_http.py:410-419** - 提取 prompt_text_ids
2. **httpserver/manager.py:201-227** - 使用预计算的 token IDs
3. **tts_encode/manager.py:133-136** - 添加 vocab offset

---

## 相关文档

- **完整模块指南**：
  - [ENCODE_MODULE_GUIDE.md](ENCODE_MODULE_GUIDE.md) - Encode 模块详解
  - [DECODE_MODULE_GUIDE.md](DECODE_MODULE_GUIDE.md) - Decode 模块详解 ⭐
- [SFT_MODE.md](SFT_MODE.md) - SFT 模式使用文档
- [SFT_MODE_DATA_FLOW.md](SFT_MODE_DATA_FLOW.md) - 完整数据流程图
- [sft_bug.md](sft_bug.md) - Bug 分析与修复记录
