# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Attention


## Common Development Commands

### Running the Server

```bash
# Quick start with default parameters
python -m light_tts.server.api_server --model_dir ./pretrained_models/CosyVoice2-0.5B-latest --load_trt True

# Using launcher script
bash launcher.sh

# Development server with custom parameters
python -m light_tts.server.api_server \
  --model_dir ./pretrained_models/CosyVoice2-0.5B-latest \
  --host localhost \
  --port 8080 \
  --load_trt True \
  --max_total_token_num 65536 \
  --max_req_total_len 32768
```

### Testing

```bash
# Zero-shot TTS test
python test/test_zero_shot.py

# Streaming TTS test
python test/test_zs_stream.py

# Performance benchmarking
python test/test_zs_speed.py

# WebSocket bidirectional streaming test
python test/test_bistream.py
```

### Docker

```bash
# Build Docker image
docker build -t light-tts:v1.0 .

# Run with GPU support
docker run -it --gpus all -p 8080:8080 --shm-size 4g -v your_local_path:/data/ light-tts:v1.0 /bin/bash
```

## High-Level Architecture

### Core Pipeline: Encode-LLM-Decode Architecture

The TTS system implements a three-stage streaming pipeline where each module runs as an independent process:

#### 1. Encode Module (`light_tts/server/tts_encode/`)
- **Purpose**: Audio feature extraction and text preprocessing
- **Functions**: Extracts speaker embeddings from prompt audio, processes input text
- **Manager**: `tts_encode/manager.py`
- **Scalability**: Multiple encode processes with semaphore control (`--encode_process_num`, `--encode_paral_num`)

#### 2. LLM Module (`light_tts/server/tts_llm/`)
- **Purpose**: Text-to-speech token generation using language models
- **Features**: LightLLM optimizations, batch processing, attention mechanisms
- **Manager**: `tts_llm/manager.py`
- **Scalability**: One process per TTS model style with parallelism control (`--gpt_paral_num`)

#### 3. Decode Module (`light_tts/server/tts_decode/`)
- **Purpose**: Converts speech tokens to audio waveforms
- **Features**: TensorRT acceleration, flow decoding, batch generation
- **Manager**: `tts_decode/manager.py`
- **Scalability**: Multiple decode processes (`--decode_process_num`, `--decode_paral_num`)

#### Data Flow Architecture
```
HTTP Request → HttpServerManager → Encode → LLM → Decode → HttpServerManager → Response
```

**Communication Mechanisms:**
- **ZMQ Pipeline**: PUSH sockets create unidirectional data flow between stages
- **Shared Memory**: Zero-copy data transfer via `ShmReqManager` and `SharedSpeechManager`
- **Process Coordination**: HttpServerManager orchestrates the entire pipeline
- **Concurrency Control**: Semaphores manage parallelism at each stage

**Request Processing:**
1. HttpServerManager receives request and allocates shared memory
2. Request index sent to Encode process
3. Extracted features flow through LLM for token generation
4. Decode converts tokens to audio waveform
5. Response streamed back through HttpServerManager

This microservices architecture enables horizontal scaling, independent module optimization, and fault isolation.

### Server Components

- **HTTP Server** (`light_tts/server/api_http.py`)
  - FastAPI-based REST API server
  - WebSocket support for bidirectional streaming
  - Health checks (`/healthz`, `/health`, `/liveness`, `/readiness`)
  - Prometheus metrics (`/metrics`)

- **HTTP Manager** (`light_tts/server/httpserver/manager.py`)
  - Orchestrates the three pipeline modules
  - Manages shared memory for speaker embeddings
  - Handles request routing and load balancing

- **Shared Memory Management** (`light_tts/server/shm_tools/`)
  - LRU cache for speaker timbre embeddings
  - Reduces memory copying across processes

### Key Configuration

- **Model Configuration**: `light_tts/static_config.py` defines supported languages
- **CLI Arguments**: `light_tts/server/api_cli.py` contains all server configuration options
- **TensorRT Support**: Use `--load_trt True` for accelerated inference

### API Endpoints

- `POST /inference_zero_shot` - Zero-shot TTS inference (streaming/non-streaming)
- `WebSocket /inference_zero_shot_bistream` - Bidirectional streaming
- `GET/POST /query_tts_model` - List available TTS models
- `GET /healthz`, `/health`, `/liveness`, `/readiness` - Health checks
- `GET /metrics` - Prometheus metrics

### Model Support

Currently supports **CosyVoice2-0.5B** model. Models should be placed in `pretrained_models/` directory:

```python
# Download models
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
```

### Performance Optimizations

#### 1. Quantization Support
- **Weight Quantization**: INT8/INT4 quantization modes (`triton_int8weight`, `triton_int4weight`, `ppl_int4weight`)
- **KV Cache Quantization**: INT8 quantization for key-value cache to reduce memory usage
- **FP8 Quantization**: Support for FP8 precision formats (`triton-fp8w8a8-b128`)
- **Mixed Precision**: Layer-wise quantization configuration through `Quantcfg` class
- **Implementation**: Located in `light_tts/common/quantization/` with custom Triton kernels

#### 2. Batch Processing Optimizations
- **Dynamic Batching**: Combines multiple requests for efficient processing
- **Token-level Batching**: Optimizes computation at token granularity
- **Prefill Batching**: Controlled via `--batch_max_tokens` to prevent OOM
- **Continuous Batching**: Supports adding new requests during processing
- **Configuration**: `--decode_max_batch_size`, `--graph_max_batch_size`

#### 3. Advanced Caching Mechanisms
- **Shared Memory Management**: Cross-process data sharing via `multiprocessing.shared_memory`
- **GPU Tensor Caching**: `CudaGraphCacheTensorManager` for efficient GPU memory reuse
- **LRU Cache**: Speaker/timbre embedding cache with eviction policies
- **KV Cache Management**: Intelligent caching with INT8 quantization support

#### 4. Hardware Accelerations
- **TensorRT Integration**: Flow decoder acceleration (`--load_trt`)
- **Custom CUDA Kernels**: Flash attention, GQA, quantized matrix multiplication
- **CUDA Graph Optimization**: Captures computation graphs for decoding phase
- **Triton Kernels**: Custom fused operations (`silu_and_mul`, attention kernels)

### Production Deployment

#### Gunicorn Production Setup
```bash
# Production deployment with Gunicorn (run.sh)
WORKERS=3  # 24GB GPU supports 3 workers for 0.5B model
gunicorn -w $WORKERS \
    -k uvicorn.workers.UvicornWorker \
    -b 0.0.0.0:8080 \
    --timeout 300 \
    --access-logfile - \
    --error-logfile - \
    wsgi_wrapper:app
```

#### WSGI Configuration
- **Wrapper**: `wsgi_wrapper.py` configures startup parameters for Gunicorn
- **Worker Safety**: Avoids `--preload` to prevent CUDA fork issues
- **Resource Management**: Each worker gets independent GPU memory
- **Configuration**: Modify `sys.argv` in `wsgi_wrapper.py` for production settings

#### Deployment Options
1. **Development**: Direct Python execution with `api_server.py`
2. **Production**: Gunicorn WSGI deployment with multi-worker scaling
3. **Containerized**: Docker with GPU support and shared memory
4. **Distributed**: Separate processes for encode/LLM/decode modules

### Git Submodules

The project uses Git submodules for:
- `cosyvoice/` - Core CosyVoice implementation
- `third_party/Matcha-TTS/` - Third-party TTS components

Initialize with: `git submodule update --init --recursive`
