# Speaker_ID 问题诊断指南

## 问题描述

使用 `speaker_id` 参数后，服务无响应，日志显示持续等待输出数据。

## 问题根源分析

### 共享内存状态机

共享内存使用 `use_mark` 字段表示状态：
- `0`: 空闲
- `1`: 已分配
- `2`: 原始音频数据已设置 (set_index_data)
- `3`: 语音特征已提取完成 (set_index_speech) ✅ **Ready**

### 问题所在

**Preset Speaker 预热流程不完整**：
- `warmup_presets()` 只调用 `alloc_speech_mem()` → `set_index_data()` → 设置 `use_mark=2`
- **缺少 `set_index_speech()` 调用** → 未设置 `use_mark=3`

**Encode 处理流程检查**：
- `speech_data_ready()` 检查 `use_mark >= 3`
- 当 `use_mark=2` 时，检查失败
- 请求被重新排队，形成死循环

## 增强日志说明

已添加详细的诊断日志到以下模块：

### 1. API 处理层 ([api_http.py:268-297](light_tts/server/api_http.py#L268-L297))

```log
INFO: Processing request with speaker_id: male
INFO: Found preset speaker config: index=0, md5=a1b2c3d4...
INFO: speaker_id 'male' speech_index=0, use_mark=2, speech_data_ready=False
WARNING: ⚠️ speaker_id 'male' speech data is NOT ready (use_mark=2, expected >= 3)
```

### 2. Preset 预热层 ([preset_speakers.py:19-70](light_tts/utils/preset_speakers.py#L19-L70))

```log
INFO: =============================================================
INFO: Starting preset speakers warmup...
INFO: =============================================================
INFO: [male] Loading preset speaker from: asset/xiangyu.wav
INFO: [male] WAV shape: (1, 38400)
INFO: [male] MD5: a1b2c3d4e5f6...
INFO: [male] Allocating shared memory...
INFO: [male] Speech index: 0, have_alloc: False
INFO: [male] Shared memory use_mark after alloc: 2 (0=free, 1=allocated, 2=data_set, 3=ready)
INFO: [male] ✅ Preset speaker loaded successfully
INFO: =============================================================
INFO: Preset speakers warmup completed. Loaded 2/2 speakers
INFO:   - male: index=0, md5=a1b2c3d4...
INFO:   - female: index=1, md5=e5f6a7b8...
INFO: =============================================================
```

### 3. Encode 处理层 ([tts_encode/manager.py:107-132](light_tts/server/tts_encode/manager.py#L107-L132))

```log
INFO: tts_encode req_id 2 speech_index 0 need_extract_speech=False, speech_data_ready=False, use_mark=2
WARNING: tts_encode req_id 2 speech_index 0 data NOT ready (use_mark=2), re-queueing...
INFO: tts_encode req_id 2 speech_index 0 need_extract_speech=False, speech_data_ready=False, use_mark=2
WARNING: tts_encode req_id 2 speech_index 0 data NOT ready (use_mark=2), re-queueing...
... (无限循环)
```

## 如何使用日志诊断问题

### 步骤 1: 重启服务并查看启动日志

观察 preset speakers 预热是否成功：
- 检查 `Shared memory use_mark after alloc` 的值
- 正常应该是 `2`（如果只有 set_index_data）
- 预期应该在 `3`（如果预热完整）

### 步骤 2: 发送测试请求

使用测试脚本 `test/tts_client_v2.py` 发送带 `speaker_id` 的请求

### 步骤 3: 观察请求处理日志

查看以下关键信息：
1. **API 层**: `speaker_id` 映射的 `speech_index` 和 `use_mark` 值
2. **Encode 层**: 是否进入死循环，`use_mark` 值是否变化

### 步骤 4: 判断问题类型

根据日志判断：

| 症状 | 原因 | 解决方案 |
|------|------|---------|
| `use_mark=2` + `speech_data_ready=False` + 死循环 | Preset 预热不完整 | 需要修复 `warmup_presets()` |
| `use_mark=0` 或 `1` | Preset 未正确加载 | 检查音频文件路径 |
| `use_mark=3` + 正常处理 | 无问题 | 日志已发挥作用 |

## 修复方案

### 方案 A: 完善 Preset 预热流程（推荐）

在 `warmup_presets()` 中提取语音特征：

```python
# preset_speakers.py
import torch
from cosyvoice.cli.frontend import CosyVoiceFrontEnd

# 在 warmup_presets() 函数中，alloc_speech_mem 之后添加：
if not have_alloc:
    # 加载 frontend
    frontend = httpserver_manager.frontend  # 需要暴露 frontend

    # 提取特征
    prompt_speech_16k_tensor = torch.from_numpy(prompt_speech_16k)
    model_input = frontend.frontend_zero_shot('', '', prompt_speech_16k_tensor, 16000, '')

    speech_token = model_input["llm_prompt_speech_token"].cpu().numpy()
    speech_feat = model_input["prompt_speech_feat"].squeeze(0).cpu().numpy()
    embedding = model_input["llm_embedding"].cpu().numpy()

    # 设置完整特征
    httpserver_manager.shared_speech_manager.set_index_speech(
        speech_index, speech_token, speech_feat, embedding
    )

    logger.info(f"[{speaker_id}] Speech features extracted, use_mark should be 3 now")
```

### 方案 B: 首次请求时提取特征（临时方案）

修改 API 层逻辑，让 preset speaker 的首次请求执行特征提取：

```python
# api_http.py
need_extract_speech = need_extract_speech and not have_alloc

# 修改为：
need_extract_speech = (need_extract_speech and not have_alloc) or (speaker_id and not have_alloc)
```

## 测试验证

### 预期正常日志（修复后）

```log
# 启动时
INFO: [male] Shared memory use_mark after alloc: 3 (0=free, 1=allocated, 2=data_set, 3=ready)

# 请求时
INFO: speaker_id 'male' speech_index=0, use_mark=3, speech_data_ready=True
INFO: tts_encode req_id 2 speech_index 0 need_extract_speech=False, speech_data_ready=True, use_mark=3
INFO: tts_encode req_id 2 using cached speech index 0
INFO: Send:    tts_encode     | req_id 2 | ... to tts_llm | with speech
```

## 下一步行动

1. **使用当前增强日志重新运行服务**
2. **收集完整日志**（从启动到请求卡住）
3. **确认 `use_mark` 值**
4. **根据分析结果应用对应的修复方案**

## 相关文件

- [light_tts/server/api_http.py](light_tts/server/api_http.py)
- [light_tts/utils/preset_speakers.py](light_tts/utils/preset_speakers.py)
- [light_tts/server/tts_encode/manager.py](light_tts/server/tts_encode/manager.py)
- [light_tts/server/core/objs/shm_speech_manager.py](light_tts/server/core/objs/shm_speech_manager.py)
