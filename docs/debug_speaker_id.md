# Speaker_ID 问题诊断指南

## 问题状态

✅ **已修复** - 详见下方"修复方案"章节

## 问题描述

使用 `speaker_id` 参数后，服务无响应，日志显示持续等待输出数据。

**症状日志**：
```log
WARNING: tts_encode req_id 0 speech_index 0 data NOT ready (use_mark=2), re-queueing...
INFO: tts_encode req_id 0 speech_index 0 need_extract_speech=False, speech_data_ready=False, use_mark=2
... (无限循环)
```

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

### ✅ 已实施的修复（方案 A：完善 Preset 预热流程）

**修改文件**：
1. [light_tts/utils/preset_speakers.py](light_tts/utils/preset_speakers.py) - 添加 frontend 参数和特征提取逻辑
2. [light_tts/server/api_http.py:108](light_tts/server/api_http.py#L108) - 传递 frontend 到 warmup_presets

**修复详情**：

#### 1. preset_speakers.py 修改

```python
# 添加 torch 导入
import torch

# 修改函数签名，添加 frontend 参数
def warmup_presets(httpserver_manager, frontend=None):
    # ... 原有代码 ...

    if not have_alloc:
        # 如果提供了 frontend，使用它提取特征
        if frontend is not None:
            # 将 numpy 数组转换为 torch tensor
            prompt_speech_16k_tensor = torch.from_numpy(prompt_speech_16k)

            # 调用 frontend 提取特征
            model_input = frontend.frontend_zero_shot(
                '', '', prompt_speech_16k_tensor, 16000, ''
            )

            speech_token = model_input["llm_prompt_speech_token"].cpu().numpy()
            speech_feat = model_input["prompt_speech_feat"].squeeze(0).cpu().numpy()
            embedding = model_input["llm_embedding"].cpu().numpy()

            # 设置完整的语音特征（这会将 use_mark 设置为 3）
            httpserver_manager.shared_speech_manager.set_index_speech(
                speech_index, speech_token, speech_feat, embedding
            )

            # 验证状态
            use_mark_after = httpserver_manager.shared_speech_manager.use_marks.arr[speech_index]
            logger.info(f"[{speaker_id}] ✅ Speech features extracted, use_mark: {use_mark_after} (expected: 3)")
```

#### 2. api_http.py 修改

```python
# 修改 warmup_presets 调用，传递 frontend
self.preset_speakers = warmup_presets(self.httpserver_manager, frontend=self.frontend)
```

### 修复效果

**修复前**：
```log
INFO: [male] Shared memory use_mark after alloc: 2 (0=free, 1=allocated, 2=data_set, 3=ready)
WARNING: tts_encode req_id 0 speech_index 0 data NOT ready (use_mark=2), re-queueing...
... (死循环)
```

**修复后（预期）**：
```log
INFO: [male] Shared memory use_mark after alloc: 2 (0=free, 1=allocated, 2=data_set, 3=ready)
INFO: [male] Extracting speech features...
INFO: [male] Using provided frontend for feature extraction
INFO: [male] ✅ Speech features extracted successfully, use_mark: 3 (expected: 3)
INFO: [male] ✅ Preset speaker loaded successfully

# 请求时
INFO: speaker_id 'male' speech_index=0, use_mark=3, speech_data_ready=True
INFO: tts_encode req_id 0 speech_index 0 need_extract_speech=False, speech_data_ready=True, use_mark=3
INFO: tts_encode req_id 0 using cached speech index 0
INFO: Send:    tts_encode     | req_id 0 | ... to tts_llm | with speech
```

## 测试验证

### 验证步骤

1. **重启服务**：
   ```bash
   python -m light_tts.server.api_server --model_dir ./pretrained_models/CosyVoice2-0.5B-latest
   ```

2. **观察启动日志**，确认特征提取成功：
   ```log
   INFO: [male] Extracting speech features...
   INFO: [male] Using provided frontend for feature extraction
   INFO: [male] ✅ Speech features extracted successfully, use_mark: 3 (expected: 3)
   INFO: [male] ✅ Preset speaker loaded successfully
   INFO: Preset speakers warmup completed. Loaded 2/2 speakers
   ```

3. **运行测试脚本**：
   ```bash
   python test/tts_client_v2.py
   ```

4. **检查请求处理日志**，确认正常流程：
   ```log
   INFO: Processing request with speaker_id: male
   INFO: speaker_id 'male' speech_index=0, use_mark=3, speech_data_ready=True
   INFO: tts_encode req_id 0 speech_index 0 need_extract_speech=False, speech_data_ready=True, use_mark=3
   INFO: tts_encode req_id 0 using cached speech index 0
   INFO: Send:    tts_encode     | req_id 0 | ... to tts_llm | with speech
   ```

5. **验证音频输出**：
   - 检查 `assets/` 目录下生成的 WAV 文件
   - 播放音频，确认质量和内容正常

## 相关文件

- [light_tts/server/api_http.py](light_tts/server/api_http.py)
- [light_tts/utils/preset_speakers.py](light_tts/utils/preset_speakers.py)
- [light_tts/server/tts_encode/manager.py](light_tts/server/tts_encode/manager.py)
- [light_tts/server/core/objs/shm_speech_manager.py](light_tts/server/core/objs/shm_speech_manager.py)
