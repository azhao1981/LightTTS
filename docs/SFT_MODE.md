# SFT 模式使用说明

## 概述

SFT (Supervised Fine-Tuning) 模式是一种优化的语音合成模式，通过使用预训练的音色 embedding，无需提供参考音频，实现更高效的推理。

## 核心特性

- ✅ **命名约定**：通过 `spk_id` 的 `_sft` 后缀区分模式
- ✅ **向后兼容**：不影响现有 Zero-Shot 功能
- ✅ **性能优化**：数据传输减少 50%（12字段→4字段）
- ✅ **最小化改动**：仅约 100 行代码

## 使用方法

python -m light_tts.server.api_server \
  --model_dir ./pretrained_models/CosyVoice2-0.5B-finetune-v1 \
  --load_trt True

python test/test_sft_integration.py

### API 调用

SFT 模式复用现有的 `/inference_zero_shot` 端点，通过 `spk_id` 后缀自动识别模式。

#### Zero-Shot 模式（现有功能）

```bash
curl -X POST http://localhost:8080/inference_zero_shot \
  -F "tts_text=你好世界" \
  -F "spk_id=female" \
  -o test_zero_shot.wav
```

#### SFT 模式（新功能）

```bash
curl -X POST http://localhost:8080/inference_zero_shot \
  -F "tts_text=你好世界" \
  -F "spk_id=female_test_sft" \
  -o test_sft.wav
```

**注意**：SFT 模式**仅支持 spk2info.pt 中的预训练说话人**，因为这些说话人有完整的缓存数据（embedding、prompt_text、llm_prompt_speech_token 等）。

**支持的 SFT 说话人 ID**（需要在 spk2info.pt 中存在）：
- `female_test_sft`（对应 `spk2info.pt` 中的 `female_test`）
- `male1_trained_sft`（对应 `spk2info.pt` 中的 `male1_trained`）
- 其他 spk2info.pt 中存在的音色

**不支持**：
- ❌ `female_sft`、`male_sft`、`male2_sft`（这些音色仅在 voices.yaml 中，缺少 SFT 所需的 embedding 缓存数据）

### Python 调用示例

```python
import requests

# SFT 模式
response = requests.post(
    "http://localhost:8080/inference_zero_shot",
    data={
        "tts_text": "你好，这是 SFT 模式测试。",
        "spk_id": "female_test_sft",  # spk2info.pt 中的音色 + _sft 后缀
        "stream": "false"
    }
)

if response.status_code == 200:
    with open("output.wav", "wb") as f:
        f.write(response.content)
```

## 命名规则

| 模式 | spk_id 格式 | 示例 | 数据来源 |
|------|------------|------|---------|
| Zero-Shot | `{voice_name}` | `female`, `male`, `female_test`, `male1_trained` | voices.yaml + spk2info.pt |
| SFT | `{voice_name}_sft` | `female_test_sft`, `male1_trained_sft` | **仅 spk2info.pt** |

**重要区别**：
- **Zero-Shot**：支持 voices.yaml 中的所有预设音色（`female`, `male`, `male2` 等）+ spk2info.pt 中的音色
- **SFT**：**仅支持 spk2info.pt 中存在的预训练说话人**（需要有完整的 embedding 缓存数据）

## 技术细节

> 💡 **想要完整的数据流程图？** 查看 [SFT_MODE_DATA_FLOW.md](SFT_MODE_DATA_FLOW.md) 了解三阶段流水线架构和三个修复点的详细位置。

### LLM Embedding Table 结构

#### 什么是 LLM Embedding Table？

**LLM Embedding Table** 是 CosyVoice2 LLM 模型（基于 Transformer 架构）的 embedding 层的词汇表结构，**不是 spk2info.pt 的内容**。

这是 CosyVoice2 特有的设计，用于在同一个模型中处理文本和语音：

```
模型 embedding 层索引:
[0, vocab_size-1]      → text tokens（文本 token）
[vocab_size]           → sos_eos（序列开始/结束标记）
[vocab_size+1]         → task_id（任务 ID）
[vocab_size+2, ...]    → speech tokens（语音 token，Codec 编码的离散单元）
```

**关键区别**：
- **标准文本 LLM**（如 GPT）：只有 text tokens `[0, vocab_size-1]`
- **CosyVoice2 LLM**：扩展了特殊 tokens（sos_eos、task_id、speech tokens），支持端到端的 TTS 任务

**代码来源**（`httpserver/manager.py:55-57`）：
```python
self.model_config = configs["llm"].llm.model.model.config
self.vocab_size = self.model_config.vocab_size  # 基础词汇表大小
self.sos_eos = self.vocab_size                   # 序列标记
self.task_id = self.vocab_size + 1               # 任务 ID
```

#### spk2info.pt 与 Embedding Table 的关系

**spk2info.pt** 是预处理的缓存数据文件，不包含 embedding table 本身，而是存储了需要查询 embedding table 的索引：

| spk2info.pt 字段 | 内容 | 需要的处理 |
|-----------------|------|-----------|
| `prompt_text` | 参考文本的 token IDs | 直接使用（已在 [0, vocab_size-1] 范围） |
| `llm_prompt_speech_token` | 参考音频的 speech token IDs | **必须加 `vocab_size + 2` 偏移** |
| `prompt_speech_feat` | 音频特征 | 用于 decode 阶段 |
| `llm_embedding` | 说话人 embedding | 直接传递给 LLM |

**为什么需要 vocab offset？**

`llm_prompt_speech_token` 存储的是 **0-based 的原始 Codec token ID**（范围 [0, codec_vocab_size)），但 CosyVoice2 LLM 的 embedding table 将 speech tokens 放在 `[vocab_size+2, ...]` 区域。因此需要加上偏移量才能正确查找：

```python
# 错误：直接使用 0-based ID
audio_ids = speech_token.flatten().tolist()  # 查找到 text tokens 区域 ❌

# 正确：加上偏移量
speech_token_offset = (speech_token + self.vocab_size + 2)
audio_ids = speech_token_offset.flatten().tolist()  # 查找到 speech tokens 区域 ✅
```

### 数据流程

```mermaid
graph TD
    subgraph "预处理阶段 (由 Codec 完成)"
        A[真实音频 WAV] -->|编码/量化| B(Speech Tokens)
        B -->|存入| C[spk2info.pt]
        style B fill:#f9f,stroke:#333
    end

    subgraph "推理阶段 (由 LLM 完成)"
        D[输入文字 Text] -->|LLM 推理| E(预测出的 Speech Tokens)
        C -->|作为提示/参考| E
        E -->|解码| F[合成的音频 WAV]
        style E fill:#f9f,stroke:#333
    end
```
### 正确的 LLM 输入结构

CosyVoice2 LLM 接收的完整 token 序列结构：

```
[sos_eos] + [prompt_text_ids] + [text_ids] + [task_id] + [speech_tokens]
   ↓              ↓                    ↓            ↓              ↓
vocab_size    [0, vocab_size)    [0, vocab_size)  vocab_size+1  [vocab_size+2, ...)
              参考文本 token        用户文本 token               语音 Codec token
```

**每个部分的含义**：
1. **sos_eos** (`vocab_size`)：序列开始标记（也是结束标记）
2. **prompt_text_ids**：参考文本的 token IDs（来自 spk2info 或实时编码）
3. **text_ids**：用户要合成的目标文本的 token IDs
4. **task_id** (`vocab_size+1`)：任务类型标识（zero-shot/SFT 等）
5. **speech_tokens**：语音 Codec 离散单元（从参考音频提取，需要加 vocab offset）

**代码构建**（`httpserver/manager.py:211`）：
```python
prompt_ids = list(chain(
    [self.sos_eos],        # [vocab_size]
    prompt_text_ids,       # 参考文本 tokens
    text_ids,              # 用户文本 tokens
    [self.task_id]         # [vocab_size+1]
))
# speech_tokens 通过 req.speech_token 传递，已加 offset
```

### Bug 修复记录 (v1.0.1)

#### 问题发现

SFT 模式使用 `spk2info.pt` 中的缓存数据（类似 zero-shot 格式）时，生成乱码音频。

#### 根因分析

两个关键 Bug：

1. **缺少 vocab offset** (`tts_encode/manager.py:133`)
   - **现象**：SFT 模式的 `speech_token` 直接使用原始 0-based ID
   - **后果**：在 embedding table 中查找到了 text tokens 区域，而非 speech tokens 区域
   - **修复**：添加 `vocab_size + 2` 偏移，与 zero-shot 模式一致

2. **缺少 prompt_text** (`api_http.py`)
   - **现象**：SFT 模式将 `prompt_text` 设为空字符串，但 `spk2info.pt` 中有 89 个预计算的 token IDs
   - **后果**：LLM 输入结构不完整，缺少参考文本部分
   - **修复**：从 `spk2info` 提取 `prompt_text` token IDs 并传递给流水线

#### 修复代码

**1. tts_encode/manager.py (第 133-137 行)**
```python
# 修复前
audio_ids = speech_token.flatten().tolist()

# 修复后
if speech_token.size > 0:
    # Add vocab_size + 2 offset for correct embedding lookup
    speech_token_offset = (speech_token + self.vocab_size + 2)
    audio_ids = speech_token_offset.flatten().tolist()
else:
    audio_ids = []
```

**2. api_http.py (第 325-334 行)**
```python
# 从 spk2info 提取 prompt_text token IDs
prompt_text_tensor = spk_info.get('prompt_text', torch.tensor([]))
if prompt_text_tensor.numel() > 0:
    prompt_text_ids = prompt_text_tensor.flatten().tolist()
    logger.info(f"SFT mode: extracted prompt_text_ids from spk2info, len={len(prompt_text_ids)}")
else:
    prompt_text_ids = []
```

**3. httpserver/manager.py (第 201-227 行)**
```python
# 使用预计算的 prompt_text_ids（如果提供）
prompt_text_ids = request_dict.get("prompt_text_ids")
if prompt_text_ids is not None:
    logger.info(f"req_id {request_id}: using pre-computed prompt_text_ids, len={len(prompt_text_ids)}")
else:
    prompt_text_ids = await self._async_encode(request_dict["prompt_text"])
```

#### 修复前后对比

| 项目 | 修复前 | 修复后 |
|------|--------|--------|
| speech_token offset | 无（0-based）| + vocab_size + 2 |
| prompt_text | 空字符串 | 从 spk2info 提取 token IDs |
| prompt_token_pad | 0 | 正确计算 padding |
| embedding 查表 | 错误位置（text tokens）| 正确位置（speech tokens）|
| semantic_len 日志 | 固定 0 | 实际值 |

#### 技术文档

- Bug 分析与修复记录：[docs/sft_bug.md](sft_bug.md)

### 数据流对比

#### Zero-Shot 模式
```
Request → API → Encode → 提取特征 → LLM → Decode → Response
                     ↓
              [speech_token, speech_feat, embedding]
                     ↓
              共享内存缓存
```

#### SFT 模式
```
Request → API → Encode → 直接跳过 → LLM → Decode → Response
                           ↓
                   使用 spk2info[spk_id]['embedding']
                           ↓
                   无需共享内存存储
```

### 字段对比

| 字段 | Zero-Shot | SFT |
|------|-----------|-----|
| `spk_id` | `"female"` | `"female_sft"` |
| `speech_index` | `0, 1, 2, ...` | 有效索引（复用现有机制） |
| `semantic_len` | `382` (语音token数) | 实际值（从 spk2info 提取） |
| `need_extract_speech` | `False` (预设音色已缓存) | `False` (embedding 已在 API 层提取) |
| `prompt_text` | `"参考文本"` | 从 spk2info 提取的 token IDs |
| `prompt_text_ids` | 实时编码 | **预计算（从 spk2info）** |
| LLM 输入字段 | **12 个** | **4 个** |

## 测试

### 运行集成测试

```bash
# Python 集成测试
python test/test_sft_integration.py
```

### 运行端到端测试

```bash
# Bash 端到端测试
bash test/test_sft_e2e.sh
```

### 手动测试

```bash
# 1. 启动服务器
python -m light_tts.server.api_server \
  --model_dir ./pretrained_models/CosyVoice2-0.5B-finetune-v1 \
  --load_trt True

# 2. 测试 Zero-Shot 模式
curl -X POST http://localhost:8080/inference_zero_shot \
  -F "tts_text=你好世界" \
  -F "spk_id=female_test" \
  -o test_zs.wav

# 3. 测试 SFT 模式
curl -X POST http://localhost:8080/inference_zero_shot \
  -F "tts_text=你好世界" \
  -F "spk_id=female_test_sft" \
  -o test_sft.wav

# 4. 查看日志，确认看到 "SFT mode detected"
```

## 日志标记

SFT 模式有以下日志标记：

- API 层：`SFT mode: extracted embedding for spk_id=..., base_spk_id=...`
- Encode 层：`SFT mode: req_id ..., spk_id=..., embedding shape=...`
- 发送到 LLM：`Send: ... | mode=SFT to tts_llm`

## 错误处理

### 无效的 SFT spk_id（使用 voices.yaml 中的音色）

```json
{
  "message": "Invalid SFT spk_id 'female_sft'. Base spk_id 'female' only in voices.yaml, not in spk2info.pt. SFT mode requires speakers with cached embedding data. Please use speakers from spk2info.pt like 'female_test_sft', 'male1_trained_sft'"
}
```

**常见错误**：尝试使用 `female_sft`、`male_sft`、`male2_sft` 等仅在 voices.yaml 中的音色

### 基础音色不存在

```json
{
  "message": "Invalid SFT spk_id 'unknown_sft'. Base spk_id 'unknown' not found in spk2info.pt. Available SFT speakers: ['female_test', 'male1_trained'] (add '_sft' suffix)"
}
```

**排查步骤**：
1. 检查 spk2info.pt 是否包含对应的基础音色（去掉 `_sft` 后缀）
2. 确保该音色在 spk2info.pt 中有完整的 embedding 数据
3. 如果音色在 voices.yaml 但不在 spk2info.pt，请使用 Zero-Shot 模式（不加 `_sft` 后缀）

## 性能指标

| 指标 | Zero-Shot | SFT | 改善 |
|------|-----------|-----|------|
| LLM 输入字段 | 12 | 4 | -67% |
| 共享内存 | 需要 | 不需要 | -100% |
| 预期延迟 | 基准 | 更快 | ~15-25% |

## 实现细节

### 修改的文件

1. **API 层** ([`api_http.py`](../light_tts/server/api_http.py))
   - 添加 SFT 模式检测：`is_sft = spk_id.endswith('_sft')`
   - **调用 `frontend_sft()` 获取 embedding**
   - **将 embedding 存储到共享内存**
   - 验证基础 spk_id（去掉 `_sft` 后缀）
   - 设置 SFT 模式参数：`speech_index`, `semantic_len=0`

2. **SpeakerManager** ([`speaker_manager.py`](../light_tts/server/speaker_manager.py))
   - 修改 `is_valid_spk_id` 支持 `_sft` 后缀验证

3. **Req 对象** ([`req.py`](../light_tts/server/core/objs/req.py))
   - 添加 `spk_id` 属性存储（Python 属性，不修改 ctypes 结构体）

4. **Encode Manager** ([`tts_encode/manager.py`](../light_tts/server/tts_encode/manager.py))
   - 添加 SFT 模式分支
   - **从共享内存读取 embedding**（API 层已存储）
   - 验证 embedding 数据已准备好
   - 发送到 LLM（semantic_len=0）

### 关键实现：embedding 传递

**问题**：如何在流水线架构中传递 embedding？

**解决方案**：
1. **API 层**调用 `frontend_sft(tts_text, base_spk_id)` 获取 model_input
2. **提取 embedding**并存储到共享内存（复用 `SharedSpeechManager.set_index_speech()`）
3. **Encode Manager**从共享内存读取 embedding 并验证
4. **LLM/Decode** 通过 speech_index 访问 embedding

**代码示例**：
```python
# API 层：调用 frontend_sft 并存储 embedding
model_input = g_objs.frontend.frontend_sft(tts_text, base_spk_id)
llm_embedding = model_input['llm_embedding'].cpu().numpy()

# 存储到共享内存（speech_token 和 speech_feat 为空）
empty_speech_token = np.array([], dtype=np.int32)
empty_speech_feat = np.array([], dtype=np.float32).reshape(0, 80)
g_objs.httpserver_manager.shared_speech_manager.set_index_speech(
    speech_index, empty_speech_token, empty_speech_feat, llm_embedding
)

# Encode Manager：从共享内存读取
speech_token, speech_feat, embedding = self.shared_speech_manager.get_index_speech(speech_index)
```

### 代码量统计

**v1.0.0 实现**：
- API 层：~70 行（包含 frontend_sft 调用和 embedding 存储）
- SpeakerManager：~10 行
- Req 对象：~3 行
- Encode Manager：~50 行（包含 embedding 读取和验证）
- **总计：约 130 行**

**v1.0.1 Bug 修复**：
- tts_encode/manager.py：+45 行（vocab offset + prompt_token_pad）
- api_http.py：+11 行（提取 prompt_text_ids）
- httpserver/manager.py：+8/-1 行（使用预计算 token IDs）
- **修复总计：约 64 行**

## 前置条件

### 模型要求

#### Zero-Shot 模式
- `voices.yaml` 必须包含基础音色（如 `female`, `male`, `male2`）
- 音色的参考音频文件必须存在于 `assets/` 目录

#### SFT 模式（额外要求）
- **`spk2info.pt` 必须包含目标音色的完整缓存数据**：
  - `embedding`（说话人 embedding）
  - `prompt_text`（参考文本 token IDs）
  - `llm_prompt_speech_token`（语音 token IDs，会自动加 vocab offset）
  - `prompt_speech_feat`（语音特征）
- ⚠️ **仅 spk2info.pt 中存在的音色可用于 SFT 模式**

### 验证配置

```bash
# 检查 voices.yaml 中的音色（Zero-Shot 模式）
cat voices.yaml

# 输出示例：
# voices:
#   - name: female
#   - name: male
#   - name: male2

# 检查 spk2info.pt 中的音色（SFT 模式）
python -c "import torch; spk2info = torch.load('pretrained_models/.../spk2info.pt'); print(list(spk2info.keys()))"

# 输出示例（可用于 SFT 模式）：
# ['female_test', 'male1_trained', ...]
# 使用时添加 '_sft' 后缀：female_test_sft, male1_trained_sft
```

## 常见问题

### Q: SFT 模式支持哪些说话人？

A: **仅支持 spk2info.pt 中存在的预训练说话人**，不是所有 voices.yaml 中的音色都支持 SFT 模式。

**原因**：SFT 模式需要完整的缓存数据（embedding、prompt_text、speech_token 等），这些数据只存在于 spk2info.pt 中。

**如何检查**：
```bash
python -c "import torch; spk2info = torch.load('path/to/spk2info.pt'); print(list(spk2info.keys()))"
```

**示例**：
- ✅ 支持：`female_test_sft`、`male1_trained_sft`（在 spk2info.pt 中）
- ❌ 不支持：`female_sft`、`male_sft`、`male2_sft`（仅在 voices.yaml 中）

如果音色仅在 voices.yaml 中，请使用 Zero-Shot 模式（不加 `_sft` 后缀）。

### Q: SFT 模式生成乱码怎么办？

A: **这是 v1.0.0 的已知 Bug，已在 v1.0.1 修复**。

**根本原因**：
1. speech_token 缺少 vocab offset，导致查表错误
2. 缺少 prompt_text token IDs

**解决方案**：
- 确保使用最新代码（commit fe16727 或更新）
- 查看日志确认 `semantic_len` 显示实际值而非 0
- 验证日志中出现 `using pre-computed prompt_text_ids`

**参考文档**：[docs/sft_bug.md](sft_bug.md)

### Q: 为什么需要 vocab offset？

A: CosyVoice2 的 LLM embedding table 是分段存储的：
- Text tokens 在 `[0, vocab_size-1]`
- Speech tokens 在 `[vocab_size+2, ...]`

spk2info.pt 中的 speech_token 是 0-based 原始 ID，必须加偏移才能正确查找。不加偏移会查找到 text tokens 区域，导致乱码。

### Q: prompt_text 和 prompt_text_ids 有什么区别？

A:
- **prompt_text**：文本字符串，需要前端实时编码（如 `"你好世界"`）
- **prompt_text_ids**：预计算的 token IDs 列表（如 `[123, 456, 789]`）

SFT 模式使用 spk2info 中的预计算数据，避免重复编码，提升性能。

### Q: SFT 模式和 Zero-Shot 模式有什么区别？

A: 核心区别在于数据传输量：
- **Zero-Shot**：使用完整的 spk2info（12个字段），包含 prompt 音频特征
- **SFT**：只使用 embedding（4个字段），数据量减少 50%

### Q: 为什么使用 `_sft` 后缀而不是新端点？

A: 命名约定方案的优势：
- 最小化代码改动（~100 行 vs ~200 行）
- API 语义清晰（spk_id 直接表达模式）
- 完全向后兼容
- 易于扩展（未来可添加 `_cross_lingual`, `_instruct` 等）

### Q: 如何验证 SFT 模式正常工作？

A: 查看日志：
1. API 层应显示：`🎯 SFT mode detected`
2. Encode 层应显示：`skipping speech extraction`
3. 发送到 LLM 应显示：`SFT mode (no speech)`

### Q: SFT 模式性能提升多少？

A: 预期延迟降低 15-25%，主要优化：
- 数据传输减少 50%
- 跳过特征提取步骤
- 无需共享内存存储

## 参考资料

- **完整数据流程图**：[docs/SFT_MODE_DATA_FLOW.md](SFT_MODE_DATA_FLOW.md) ⭐⭐
- **完整模块指南**：
  - [docs/ENCODE_MODULE_GUIDE.md](ENCODE_MODULE_GUIDE.md) - Encode 模块详解
  - [docs/DECODE_MODULE_GUIDE.md](DECODE_MODULE_GUIDE.md) - Decode 模块详解
- **Bug 分析与修复记录**：[docs/sft_bug.md](sft_bug.md) ⭐
- 技术分析报告：[docs/analysis/inference_sft_analysis.md](analysis/inference_sft_analysis.md)
- 实施计划：[.claude/plans/idempotent-strolling-gizmo.md](../../.claude/plans/idempotent-strolling-gizmo.md)
- 集成测试：[test/test_sft_integration.py](../../test/test_sft_integration.py)
- 端到端测试：[test/test_sft_e2e.sh](../../test/test_sft_e2e.sh)

## 更新日志

### v1.0.1 (2026-01-14) - Bug 修复

- 🐛 **修复 vocab offset 缺失**：SFT 模式 speech_token 现在正确添加 `vocab_size + 2` 偏移
- 🐛 **修复 prompt_text 缺失**：从 spk2info.pt 提取预计算的 prompt_text token IDs
- ✅ **更新 prompt_token_pad 计算**：正确处理 token padding
- ✅ **修复 semantic_len 日志**：从固定 0 改为实际值
- 📚 **新增技术文档**：添加 LLM embedding table 结构说明
- 📚 **新增 Bug 分析文档**：[docs/sft_bug.md](sft_bug.md)

**影响文件**：
- `light_tts/server/tts_encode/manager.py` (+45 行)
- `light_tts/server/api_http.py` (+11 行)
- `light_tts/server/httpserver/manager.py` (+8/-1 行)

### v1.0.0 (2025-01-14)

- ✅ 添加 SFT 模式支持
- ✅ 命名约定方案（`_sft` 后缀）
- ✅ 修改 4 个核心文件
- ✅ 添加集成测试和端到端测试
- ✅ 完整文档
