# 预设音色功能使用指南

## 功能概述

LightTTS 现在支持预设音色功能,允许在服务启动时预加载音色,并在推理时直接通过音色 ID 引用,避免每次请求都上传参考音频。

## 核心优势

- **性能提升**: 消除重复特征提取,节省约 200-500ms/请求
- **资源复用**: 利用 CosyVoice2 内部缓存机制
- **易于扩展**: 新增音色只需修改 YAML 配置,无需改代码
- **向后兼容**: 原有的动态上传音频方式仍然可用

## 配置文件

### voices.yaml 格式

```yaml
voices:
  - name: female
    audio_path: assets/wangye1.mp3
    prompt_text: "使用费的话,这一块是属于你管品牌就三万块钱。培训的话,培训费要两万。"

  - name: female_long
    audio_path: assets/wangye1.wav
    prompt_text: "使用费的话,这一块是属于你管品牌就三万块钱。培训的话,培训费要两万,然后设计费要三千,然后系统使用费要两两千块钱这样子。那这一次的话就是今年我们加盟它,这些都是减免的。"

  - name: male
    audio_path: assets/xiangyu.mp3
    prompt_text: "咱们这个项目呢,属于是投入低、回本快。而且现在加盟呢,还有一些政策上的这个优惠。"
```

**字段说明**:
- `name`: 音色唯一标识符 (推理时使用)
- `audio_path`: 参考音频文件路径 (相对于 YAML 文件或绝对路径)
- `prompt_text`: 参考音频对应的文本 (必须准确匹配)

## API 使用

### 1. 查询可用预设音色

```bash
curl http://localhost:8080/query_presets
```

**响应示例**:
```json
{
  "presets": ["female", "female_long", "male", "male_long", "male2"],
  "loaded_count": 5,
  "failed_count": 0,
  "total_count": 5
}
```

### 2. 使用预设音色进行推理

**方式 1: 表单提交**
```bash
curl -X POST http://localhost:8080/inference_zero_shot \
  -F "tts_text=你好,世界!" \
  -F "spk_id=female" \
  -F "stream=false"
```

**方式 2: Python 测试脚本**
```bash
# 查询可用音色
python test/test_presets.py --query

# 使用预设音色生成语音
python test/test_presets.py \
  --text "你好,这是测试。" \
  --spk_id female \
  --output output.wav

# 使用流式推理
python test/test_presets.py \
  --text "你好,这是测试。" \
  --spk_id male \
  --stream \
  --output output_stream.wav
```

### 3. 原有方式(动态上传)仍然可用

```bash
curl -X POST http://localhost:8080/inference_zero_shot \
  -F "tts_text=你好,世界!" \
  -F "prompt_text=希望你以后能够做的比我还好呦。" \
  -F "prompt_wav=@/path/to/prompt.wav" \
  -F "stream=false"
```

## API 参数说明

### /inference_zero_shot 端点

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `tts_text` | string | ✅ | 要合成的文本 |
| `spk_id` | string | ⚠️ | 预设音色 ID (与 `prompt_wav`+`prompt_text` 二选一) |
| `prompt_text` | string | ⚠️ | 参考文本 (使用动态上传时必填) |
| `prompt_wav` | file | ⚠️ | 参考音频 (使用动态上传时必填) |
| `stream` | boolean | ❌ | 是否流式输出 (默认: false) |
| `speed` | float | ❌ | 语速倍率 (默认: 1.0, 仅非流式有效) |
| `tts_model_name` | string | ❌ | TTS 模型名称 (默认: "CosyVoice2") |

**重要**: 使用预设音色时,`spk_id` 参数优先级更高。此时 `prompt_text` 和 `prompt_wav` 应省略或留空。

## 错误处理

### 常见错误及解决方案

#### 1. 无效的 spk_id
```json
{
  "message": "Invalid spk_id 'invalid_id'. Available presets: ['female', 'male', ...]"
}
```
**解决**: 使用 `/query_presets` 查询可用音色 ID

#### 2. 音色未加载
```json
{
  "message": "SpeakerManager not initialized"
}
```
**解决**: 检查服务启动日志,确认 `voices.yaml` 文件存在且格式正确

#### 3. 缺少参数
```json
{
  "message": "Please provide either spk_id (preset voice) or both prompt_wav and prompt_text (custom voice)"
}
```
**解决**: 提供完整的参数组合

## 服务启动日志

启动时,服务会输出音色加载日志:

```
============================================================
开始加载预设音色...
配置文件: /path/to/voices.yaml
------------------------------------------------------------
✓ 加载成功: [female] - assets/wangye1.mp3
✓ 加载成功: [female_long] - assets/wangye1.wav
✓ 加载成功: [male] - assets/xiangyu.mp3
✓ 加载成功: [male_long] - assets/xiangyu.wav
✓ 加载成功: [male2] - assets/xiangyu1.mp3
------------------------------------------------------------
音色加载完成: 成功 5 个, 失败 0 个
可用音色 ID: ['female', 'female_long', 'male', 'male_long', 'male2']
============================================================
```

## 性能对比

| 方式 | 首次推理延迟 | 内存占用 | 适用场景 |
|------|-------------|---------|---------|
| **预设音色** | ~100ms | 固定 (10-50MB) | 高频请求,固定音色 |
| **动态上传** | ~300-500ms | 按需分配 | 低频请求,自定义音色 |

## 最佳实践

1. **音色选择**: 选择 3-10 秒的清晰音频作为参考
2. **文本匹配**: `prompt_text` 必须与音频内容精确匹配
3. **音色命名**: 使用语义化命名 (如 `female_young`, `male_old`)
4. **路径管理**: 音频文件建议放在 `assets/` 目录下
5. **测试验证**: 使用 `/query_presets` 确认音色加载成功

## 故障排查

### 音色加载失败

1. 检查日志中的错误信息
2. 确认音频文件路径正确
3. 验证音频格式 (支持: WAV, MP3)
4. 检查 `prompt_text` 是否为空

### 推理速度慢

1. 确认使用了预设音色 (不是动态上传)
2. 检查 `stream=false` (非流式更快)
3. 考虑批量请求

### 音质问题

1. 更换参考音频 (选择清晰、无背景音的样本)
2. 调整 `speed` 参数
3. 尝试不同的音色 ID

## 架构说明

- **模块**: `light_tts/server/speaker_manager.py`
- **配置**: `voices.yaml` (项目根目录)
- **API 层**: 修改了 `api_http.py`
- **缓存机制**: CosyVoice2 `frontend.spk2info` 字典

## 更新日志

- **v1.0.0** (2024-01-06)
  - 初始版本
  - 支持预设音色加载
  - 添加 `/query_presets` 端点
  - 修改 `/inference_zero_shot` 支持 `spk_id` 参数
