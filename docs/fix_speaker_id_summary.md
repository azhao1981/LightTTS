# Speaker_ID 问题修复总结

## 问题

使用 `speaker_id="male"` 参数时，服务无响应，日志显示死循环：
```
WARNING: tts_encode req_id 0 speech_index 0 data NOT ready (use_mark=2), re-queueing...
```

## 根本原因

**Preset Speaker 预热不完整**：
- `warmup_presets()` 只调用 `alloc_speech_mem()` → 设置 `use_mark=2`（原始音频数据）
- 缺少语音特征提取步骤 → 未设置 `use_mark=3`（完整特征）
- Encode 模块检查 `speech_data_ready()` 要求 `use_mark >= 3`
- 检查失败后请求重新排队，形成**死循环**

## 修复方案

✅ **已在代码中实施**

### 修改文件

1. **[light_tts/utils/preset_speakers.py](../light_tts/utils/preset_speakers.py)**
   - 添加 `frontend` 参数到 `warmup_presets()` 函数
   - 在预热阶段提取完整的语音特征（speech_token, speech_feat, embedding）
   - 调用 `set_index_speech()` 将 `use_mark` 设置为 3

2. **[light_tts/server/api_http.py](../light_tts/server/api_http.py)**
   - 修改第 108 行：传递 `frontend` 到 `warmup_presets()`
   - 确保在 frontend 初始化后再调用预热

3. **增强诊断日志**（保留用于问题排查）
   - [preset_speakers.py](../light_tts/utils/preset_speakers.py): 详细的预热过程日志
   - [api_http.py](../light_tts/server/api_http.py): 请求处理状态检查
   - [tts_encode/manager.py](../light_tts/server/tts_encode/manager.py): 编码队列诊断

## 验证方法

1. 重启服务，观察启动日志：
   ```
   INFO: [male] ✅ Speech features extracted successfully, use_mark: 3 (expected: 3)
   INFO: Preset speakers warmup completed. Loaded 2/2 speakers
   ```

2. 运行测试 `python test/tts_client_v2.py`

3. 确认日志中 `use_mark=3` 且正常完成请求

## 相关文档

- 详细分析：[debug_speaker_id.md](debug_speaker_id.md)
- 测试脚本：[test/tts_client_v2.py](../test/tts_client_v2.py)
