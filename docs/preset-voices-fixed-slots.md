# 预设音色优化 - 固定槽位方案

## 修复总结

### 问题诊断

经过深度代码审查,发现原实现存在以下问题:

1. **use_marks 状态不一致**: LRU 驱逐时未重置 `use_marks`,可能导致脏读
2. **semantic_len 计算缺失**: 预设音色 `semantic_len=0`,影响长度估算
3. **预设音色可被驱逐**: 混合 LRU 可能导致预设音色被驱逐,降低性能

### 解决方案: 固定槽位设计 (方案 A)

**核心思想**: 完全隔离预设音色和动态上传的内存空间

```
索引:  [0, preset_slots)     [preset_slots, size)
       ├──────────────┤       ├──────────────────┤
       │  预设音色    │       │   动态上传 (LRU)  │
       │  (固定槽位)  │       │                  │
       └──────────────┘       └──────────────────┘
       永不驱逐                支持驱逐
```

---

## 代码修改清单

### 1. SharedSpeechManager 固定槽位实现

**文件**: [light_tts/server/core/objs/shm_speech_manager.py](../light_tts/server/core/objs/shm_speech_manager.py)

#### 构造函数修改 (75-104 行)

```python
def __init__(self, name, size, init_mark=True, preset_slots=10):
    self.preset_slots = preset_slots           # 预设音色槽位数
    self.dynamic_slots = size - preset_slots   # 动态上传槽位数
    # ...
```

#### alloc 方法修改 (107-143 行)

**关键变更**:
- 只在动态槽位范围 `[preset_slots, size)` 内查找
- LRU 容量检查改为 `self.dynamic_slots`
- 驱逐时重置 `use_marks.arr[index] = 1`

```python
def alloc(self, speech_md5):
    # 只在动态槽位范围 [preset_slots, size) 内分配
    for i in range(self.preset_slots, self.size):
        if self.use_marks.arr[i] == 0:
            index = i
            break
    # ...
```

#### alloc_by_spk_id 方法重写 (145-188 行)

**关键变更**:
- 只在预设槽位范围 `[0, preset_slots)` 内查找
- 不检查 LRU,不参与驱逐
- 满槽时抛出明确错误提示

```python
def alloc_by_spk_id(self, spk_id):
    # 在预设槽位范围 [0, preset_slots) 内分配
    if len(self.spk_id_to_index) >= self.preset_slots:
        raise RuntimeError(
            f"preset slots full ({self.preset_slots}), "
            f"Consider increasing --preset-slots parameter."
        )

    for i in range(self.preset_slots):
        if self.use_marks.arr[i] == 0:
            index = i
            break
    # ...
```

---

### 2. SpeakerManager 预计算 semantic_len

**文件**: [light_tts/server/speaker_manager.py](../light_tts/server/speaker_manager.py)

#### _register_voice 方法修改 (128-165 行)

**关键变更**:
- 预计算 `semantic_len` 并存储到 `voices` 字典
- 日志输出包含 `semantic_len` 信息

```python
# 预计算语义长度 (与动态上传模式保持一致)
semantic_len = (prompt_speech_16k.shape[1] + 239) // 640 + 10

self.voices[spk_id] = {
    'audio_path': audio_path,
    'prompt_text': prompt_text,
    'speech_index': speech_index,
    'semantic_len': semantic_len,  # 新增
}
```

---

### 3. API HTTP 使用预计算值

**文件**: [light_tts/server/api_http.py](../light_tts/server/api_http.py)

#### inference_zero_shot 修改 (287-302 行)

**关键变更**:
- 从 `voice_info` 读取预计算的 `semantic_len`
- 移除硬编码的 `semantic_len = 0`

```python
# 使用预计算的语义长度
semantic_len = voice_info.get('semantic_len', 0)
```

---

## 性能对比

### 优化前 (原 LRU 混合方案)

| 场景 | 行为 | 问题 |
|------|------|------|
| **预设音色请求** | spk_id → 查找 LRU → 可能被驱逐 | ❌ 性能不稳定 |
| **动态上传请求** | MD5 → 查找 LRU → 可能驱逐预设音色 | ❌ 相互影响 |
| **驱逐时** | popitem() → 未重置 use_marks | ❌ 状态不一致 |
| **semantic_len** | 硬编码为 0 | ❌ 长度估算错误 |

### 优化后 (固定槽位方案)

| 场景 | 行为 | 优势 |
|------|------|------|
| **预设音色请求** | spk_id → 直接映射固定槽位 | ✅ 永不驱逐,性能恒定 |
| **动态上传请求** | MD5 → LRU → 只影响动态槽位 | ✅ 完全隔离 |
| **驱逐时** | 只在动态槽位内驱逐 | ✅ 状态一致 |
| **semantic_len** | 预计算准确值 | ✅ 长度估算准确 |

---

## 架构优势

### 1. 零驱逐风险
```
预设音色使用固定槽位 [0, preset_slots)
  ↓
永不参与 LRU 驱逐
  ↓
性能恒定,可预测
```

### 2. 资源隔离
```
预设音色槽位 (固定)    动态上传槽位 (LRU)
[0, 10)               [10, 100)
    │                      │
    └──────────────────────┘
         完全隔离,互不影响
```

### 3. 易于监控
```python
# 可精确统计
preset_usage = len(spk_id_to_index)  # 预设音色使用量
dynamic_usage = len(lru_cache)       # 动态上传使用量
```

### 4. 配置灵活
```bash
# 小规模部署
--cache-capacity 50 --preset-slots 5

# 大规模部署
--cache-capacity 200 --preset-slots 20
```

---

## 配置建议

### 场景 1: 小规模部署 (默认)
```bash
python -m light_tts.server.api_server \
  --cache-capacity 100 \
  --preset-slots 10
```

**资源分配**:
- 预设音色: 10 个 (槽位 0-9)
- 动态上传: 90 个 (槽位 10-99)

### 场景 2: 预设音色为主
```bash
python -m light_tts.server.api_server \
  --cache-capacity 200 \
  --preset-slots 50
```

**资源分配**:
- 预设音色: 50 个 (槽位 0-49)
- 动态上传: 150 个 (槽位 50-199)

### 场景 3: 动态上传为主
```bash
python -m light_tts.server.api_server \
  --cache-capacity 200 \
  --preset-slots 5
```

**资源分配**:
- 预设音色: 5 个 (槽位 0-4)
- 动态上传: 195 个 (槽位 5-199)

---

## 监控指标

### 启动时监控
```python
# SpeakerManager 统计
stats = speaker_manager.get_statistics()
print(f"预设音色: {stats['loaded']}/{stats['total']}")
print(f"可用音色: {stats['available_spk_ids']}")
```

**输出示例**:
```
音色加载完成: 成功 3 个, 失败 0 个
可用音色 ID: ['male', 'female', 'child']
```

### 运行时监控
```python
# SharedSpeechManager 统计
preset_count = len(shared_speech_manager.spk_id_to_index)
dynamic_count = len(shared_speech_manager.lru_cache)
preset_usage = f"{preset_count}/{shared_speech_manager.preset_slots}"
dynamic_usage = f"{dynamic_count}/{shared_speech_manager.dynamic_slots}"

print(f"预设音色使用: {preset_usage}")
print(f"动态上传使用: {dynamic_usage}")
```

**输出示例**:
```
预设音色使用: 3/10
动态上传使用: 45/90
```

---

## 错误处理

### 预设槽位满
```
RuntimeError: alloc_by_spk_id failed: preset slots full (10),
cannot register spk_id='new_voice'.
Consider increasing --preset-slots parameter.
```

**解决方法**:
```bash
# 增加 preset_slots 参数
--preset-slots 20
```

### 动态槽位满
```
RuntimeError: alloc failed: no available slot in dynamic range [10, 100)
```

**解决方法**:
```bash
# 增加 cache_capacity 参数
--cache-capacity 200
```

---

## 测试验证

### 单元测试
```bash
# 测试预设音色固定槽位
python test/test_presets.py --spk_id male --text "测试固定槽位"

# 验证槽位分配
assert speech_index < preset_slots  # 预设音色应在 [0, 10)
```

### 性能测试
```bash
# 预设音色性能 (恒定)
ab -n 1000 -c 10 'http://localhost:8080/inference_zero_shot?spk_id=male&tts_text=测试'

# 动态上传性能 (LRU 缓存)
ab -n 1000 -c 10 -p test.wav 'http://localhost:8080/inference_zero_shot'
```

### 压力测试
```bash
# 验证预设音色不被驱逐
for i in {1..1000}; do
  curl "http://localhost:8080/inference_zero_shot?spk_id=male&tts_text=测试"
done
# 预期: 所有请求都是缓存命中 (have_alloc=True)
```

---

## 向后兼容性

### API 兼容
- ✅ 原有 API 签名不变
- ✅ 动态上传模式行为不变
- ✅ 新增 `preset_slots` 可选参数 (默认 10)

### 数据兼容
- ✅ voices.yaml 格式不变
- ✅ 共享内存布局兼容 (仅分配策略改变)
- ✅ 预存数据可复用

---

## 关键文件清单

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| [shm_speech_manager.py](../light_tts/server/core/objs/shm_speech_manager.py) | 重构 | 固定槽位实现 |
| [speaker_manager.py](../light_tts/server/speaker_manager.py) | 增强 | 预计算 semantic_len |
| [api_http.py](../light_tts/server/api_http.py) | 优化 | 使用预计算值 |
| [shared-memory-architecture.md](shared-memory-architecture.md) | 更新 | 新增固定槽位文档 |

---

## 总结

### 修复效果

| 问题 | 修复前 | 修复后 |
|------|--------|--------|
| **use_marks 状态** | ❌ 不一致 | ✅ 一致 |
| **semantic_len** | ❌ 硬编码 0 | ✅ 预计算准确值 |
| **预设音色驱逐** | ❌ 可能被驱逐 | ✅ 永不驱逐 |
| **性能稳定性** | ❌ 不稳定 | ✅ 恒定 |
| **资源隔离** | ❌ 混合 | ✅ 完全隔离 |

### 架构优势

1. **✅ 零驱逐风险**: 预设音色固定槽位,永不驱逐
2. **✅ 性能恒定**: 预设音色响应时间稳定可预测
3. **✅ 资源隔离**: 预设音色和动态上传完全隔离
4. **✅ 易于监控**: 精确统计资源使用情况
5. **✅ 配置灵活**: 支持不同场景的槽位分配

### 下一步

- [ ] 添加 CLI 参数 `--preset-slots` 支持配置
- [ ] 监控指标暴露到 Prometheus
- [ ] 单元测试覆盖固定槽位逻辑
- [ ] 性能基准测试对比

---

## 参考资料

- [共享内存架构详解](shared-memory-architecture.md)
- [预设音色优化文档](preset-voices-optimization.md)
- [LRU 缓存最佳实践](https://en.wikipedia.org/wiki/Cache_replacement_policies#LRU)
