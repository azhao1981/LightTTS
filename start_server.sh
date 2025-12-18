#!/bin/bash

# A10 Optimized LightTTS Startup Script
# Optimized for 24GB VRAM with balanced TTFA and throughput

# Environment setup
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)/cosyvoice"

# Model configuration
MODEL_DIR="./pretrained_models/CosyVoice2-0.5B"
HOST="0.0.0.0"
PORT=8080

# A10 optimized parameters
HTTPSERVER_WORKERS=2          # 2 HTTP workers for concurrency
ENCODE_PROCESS_NUM=1          # 1 encode process sufficient
DECODE_PROCESS_NUM=2          # 2 decode processes for throughput
ENCODE_PARAL_NUM=32           # Moderate parallelism
GPT_PARAL_NUM=32              # Moderate parallelism
DECODE_PARAL_NUM=2            # Allow 2 concurrent decodes
DECODE_MAX_BATCH_SIZE=2       # Batch size 2 for better utilization
MAX_TOTAL_TOKEN_NUM=32768     # Reduced for 2 workers
MAX_REQ_TOTAL_LEN=8192        # Per-request limit
BATCH_MAX_TOKENS=4096         # Control prefill batch size
GRAPH_MAX_BATCH_SIZE=4        # CUDA Graph optimization
DATA_TYPE="bfloat16"          # A10 optimized precision

# TensorRT and optimization flags
LOAD_TRT=True
DISABLE_CUDAGRAPH=False
MODE="triton_flashdecoding"   # Flash Attention for A10

echo "Starting LightTTS with A10 optimizations..."
echo "Model: $MODEL_DIR"
echo "Precision: $DATA_TYPE"
echo "HTTP Workers: $HTTPSERVER_WORKERS"
echo "Decode Processes: $DECODE_PROCESS_NUM"

# Start server with Gunicorn
gunicorn -w $HTTPSERVER_WORKERS \
    -k uvicorn.workers.UvicornWorker \
    -b $HOST:$PORT \
    --timeout 300 \
    --access-logfile - \
    --error-logfile - \
    --preload \
    wsgi_wrapper:app \
    --env MODEL_DIR=$MODEL_DIR \
    --env DATA_TYPE=$DATA_TYPE \
    --env ENCODE_PROCESS_NUM=$ENCODE_PROCESS_NUM \
    --env DECODE_PROCESS_NUM=$DECODE_PROCESS_NUM \
    --env ENCODE_PARAL_NUM=$ENCODE_PARAL_NUM \
    --env GPT_PARAL_NUM=$GPT_PARAL_NUM \
    --env DECODE_PARAL_NUM=$DECODE_PARAL_NUM \
    --env DECODE_MAX_BATCH_SIZE=$DECODE_MAX_BATCH_SIZE \
    --env MAX_TOTAL_TOKEN_NUM=$MAX_TOTAL_TOKEN_NUM \
    --env MAX_REQ_TOTAL_LEN=$MAX_REQ_TOTAL_LEN \
    --env BATCH_MAX_TOKENS=$BATCH_MAX_TOKENS \
    --env GRAPH_MAX_BATCH_SIZE=$GRAPH_MAX_BATCH_SIZE \
    --env LOAD_TRT=$LOAD_TRT \
    --env DISABLE_CUDAGRAPH=$DISABLE_CUDAGRAPH \
    --env MODE=$MODE

echo "LightTTS server started on $HOST:$PORT"