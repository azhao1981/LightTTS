# Light TTS 文档

## 项目概述

Light TTS 是一个高性能的文本转语音（TTS）系统，采用 Encode-LLM-Decode 三阶段流水线架构，支持零样本语音合成和实时流式处理。

## 文档目录

### 核心文档
- **[架构设计](./arch.md)** - 系统架构、性能优化和部署方式
- **[参数配置](./args.md)** - 启动参数详细说明和配置指南
- **[生产部署](./opz-conc.md)** - 生产环境配置和GPU利用率优化

### 快速开始

#### 1. 开发环境启动
```bash
python -m light_tts.server.api_server \
  --model_dir ./pretrained_models/CosyVoice2-0.5B-latest \
  --load_trt True \
  --host localhost \
  --port 8080
```

#### 2. 生产环境启动
```bash
# 使用内置配置
python wsgi_wrapper.py

# 或使用运行脚本
bash run.sh
```

## 核心特性

### 🚀 性能优化
- **量化支持**: INT8/INT4权重量化、KV Cache量化、FP8精度
- **批处理优化**: 动态批处理、Token级批处理、连续批处理
- **缓存机制**: 共享内存管理、GPU张量缓存、LRU策略
- **硬件加速**: TensorRT集成、自定义CUDA内核、CUDA Graph优化

### 🏗️ 架构设计
- **微服务架构**: Encode-LLM-Decode三阶段独立进程
- **零拷贝通信**: 共享内存 + ZMQ消息队列
- **水平扩展**: 各阶段支持独立扩展
- **流式处理**: 支持实时双向流式TTS

### 🔧 部署选项
- **开发模式**: 直接Python启动
- **生产模式**: Gunicorn多Worker部署
- **容器化**: Docker GPU支持
- **分布式**: 多实例负载均衡

## API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/inference_zero_shot` | POST | 零样本TTS推理（流式/非流式） |
| `/inference_zero_shot_bistream` | WebSocket | 双向流式TTS |
| `/query_tts_model` | GET/POST | 查询可用TTS模型 |
| `/healthz`, `/health` | GET | 健康检查 |
| `/metrics` | GET | Prometheus指标 |

## 模型支持

目前支持 **CosyVoice2-0.5B** 模型：
- 零样本语音克隆
- 多语言支持
- 情感合成
- 高质量音频输出

## 性能参考

### 硬件配置
- **24GB GPU**: 支持3个Worker处理0.5B模型
- **显存占用**: 单实例约7GB
- **并发能力**: 支持多路实时流式处理

### 优化建议
- 启用TensorRT: `--load_trt True`
- 调整token容量: `--max_total_token_num 65536`
- 多进程部署: 增加decode进程数量

## 更多资源

- [GitHub仓库](https://github.com/your-org/light-tts)
- [问题反馈](https://github.com/your-org/light-tts/issues)
- [更新日志](./CHANGELOG.md)