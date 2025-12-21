# LightTTS A10 高性能优化报告

## 执行摘要

**设备**: NVIDIA A10 (24GB VRAM, Ampere Architecture)
**模型**: CosyVoice2-0.5B
**目标**: 单进程算子级优化提升 TTFA (首响) 和 RTF (实时率)

### 优化成果预览

| 优化项 | 理论收益 | 实现方式 | 风险等级 |
|--------|----------|----------|----------|
| **BF16 精度** | 稳定性↑, 无需 Loss Scaler | 1 行代码修改 | 低 ⭐⭐ |
| **Flow 步数优化** | RTF ↓ 30-50% | 3 处参数化修改 | 中 ⭐⭐⭐ |
| **TensorRT** | 解码加速 | CLI 参数 | 低 ⭐ |
| **量化 (可选)** | 内存↓, 吞吐量↑ | CLI 参数 | 中 ⭐⭐ |

---

## 1. 参数路径深度审计

### 1.1 CLI 参数映射
```bash
# 关键命令行参数
--mode triton_flashdecoding     # LLM 推理模式
--load_trt True                 # Flow 解码器 TensorRT 加载
--graph_max_batch_size 32       # CUDA 图捕获批大小
--batch_max_tokens 8192         # 预填充批限制
```

### 1.2 内部参数流向
```
LightTTS Server Launch
└── api_cli.py (CLI 解析)
    └── HttpServerManager
        └── RouterManager(args.mode, args.load_trt)
            └── ModelRpcClient.init_model()
                └── ContinuesBatchBackend(data_type="float16")
                    └── CosyVoice2TpPartModel
                        └── Flow.py (n_timesteps=10 硬编码)
```

### 1.3 核心推理文件定位
- **LLM 核心**: `light_tts/server/tts_llm/model_infer/mode_backend/continues_batch/impl.py`
- **解码核心**: `light_tts/server/tts_decode/model_infer/model_rpc.py`
- **Flow 步数**: `cosyvoice/cosyvoice/flow/flow.py:142, 276` (`n_timesteps=10`)

---

## 2. 内核级优化分析

### 2.1 精度优化 (BF16 vs FP16)

#### 现状
- ✅ **底层已支持**：`basemodel.py:188-192` 支持 `bfloat16`
- ❌ **默认 FP16**：`model_rpc.py:75` 默认 `float16`
- ✅ **A10 兼容**：Ampere 架构原生支持 BF16

#### 优势对比
| 特性 | FP16 (默认) | BF16 (优化后) |
|------|-------------|---------------|
| 范围 | 65,504 | ~3.4e38 |
| 精度 | 10^-3 | 7.8e-3 |
| Loss Scaler | 需要 | 不需要 ✅ |
| A10 稳定性 | 中 | 高 ✅ |

---

### 2.2 Flow Matching 步数调优

#### 现状
```python
# cosyvoice/flow/flow.py:142
feat, flow_cache = self.decoder(
    mu=h.transpose(1, 2).contiguous(),
    mask=mask.unsqueeze(1),
    spks=embedding,
    cond=conds,
    n_timesteps=10,  # ❌ 硬编码，无法调整
    prompt_len=mel_len1,
    cache=flow_cache
)
```

#### A10 可行性分析
- **默认**: 10 步 (平衡质量)
- **A10 优化**: 5-8 步
- **预期收益**: RTF ↓ 20-50%，音质轻微下降

---

### 2.3 Attention 机制 (SDPA)

#### 现状
```python
# cosyvoice/transformer/attention.py:196
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # 朴素
attn = torch.softmax(scores, dim=-1)  # 无融合优化
x = torch.matmul(p_attn, value)
```

#### 结论
**⚠️ 框架不支持**：修改可能影响模型权重兼容性，代价过高。
**建议**：不采用，除非模型完全重训。

---

### 2.4 量化支持 (INT8/INT4)

#### CLI 原生支持
```bash
python -m light_tts.server.api_server \
  --mode triton_int8kv triton_int4weight \
  --load_trt True
```

#### 结论
✅ **可用但需验证**：`--mode` 参数原生支持量化，无需代码修改。

---

## 3. 实施方案

### 3.1 修改清单

| 文件 | 修改内容 | 风险 | 回滚方案 |
|------|----------|------|----------|
| `model_rpc.py` | `data_type` 默认值改为 `bfloat16` | 低 | 直接回退 |
| `flow.py` | `n_timesteps` 参数化 | 中 | 注释回退 |
| `model.py` | `token2wav` 支持动态步数 | 中 | 默认值保护 |
| `httpserver/manager.py` | 透传 `speed` → `flow_steps` | 低 | 添加兼容层 |

---

## 4. 验证方案

### 4.1 性能验证
```bash
# 内存监控
nvidia-smi -l 1

# 基准测试 (对比 FP16/BF16)
python test/test_zs_speed.py --dtype bfloat16 --steps 5
python test/test_zs_speed.py --dtype float16 --steps 10
```

### 4.2 质量验证
```bash
# 音质和一致性测试
python test/test_zero_shot.py
# 检查 MOS 分值差异 < 5%
```

### 4.3 稳定性测试
```bash
# 24h 长期压力测试
python test/test_bistream.py --duration 1440
```

---

## 5. 最终建议的 A10 启动命令

```bash
cd ~/tts && source .envrc
export FINETURED_COSYVOICE2_MODEL_PATH="/root/.cache/modelscope/hub/models/azhao2050/CosyVoice2-0.5B-finetune-v1"

# 基于补丁后的优化配置
python -m light_tts.server.api_server \
  --model_dir "$FINETURED_COSYVOICE2_MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8080 \
  --load_trt True \
  --mode triton_flashdecoding \
  --max_total_token_num 65536 \
  --max_req_total_len 32768 \
  --graph_max_batch_size 32 \
  --batch_max_tokens 8192
```

**启动参数说明**:
- ✅ **BF16**: 源码补丁生效后自动启用
- ✅ **Flow 步数**: 通过模型初始化配置或后续 CLI 扩展
- ✅ **TensorRT**: 已启用
- ⏳ **量化**: 按需添加 `--mode triton_int8kv triton_int4weight`

---

---

## 6. 源码补丁应用指南

### 补丁应用顺序
1. **BF16 精度升级** (model_rpc.py)
2. **Flow 步数参数化** (flow.py, model.py)
3. **配置透传** (manager.py)

### 创建时间
`2025-12-18 A10 优化专版`

---

**文档版本**: v1.2
**维护者**: HPC Optimization Engineer
**下次评审**: 2026-01-18 (30天周期)
