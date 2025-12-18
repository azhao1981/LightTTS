#!/bin/bash

# A10 Optimized LightTTS Startup Script
# Optimized for 24GB VRAM with balanced TTFA and throughput

# Environment setup
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)/cosyvoice"

# Check if prompt audio is provided
if [ ! -f "test/prompt.wav" ]; then
    echo "Warning: test/prompt.wav not found. Please provide a prompt audio file for testing."
fi

echo "Starting LightTTS with A10 optimizations..."

# Use environment variables to override default config
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8080}"
export MODEL_DIR="${MODEL_DIR:-./pretrained_models/CosyVoice2-0.5B}"

# A10 optimized parameters (can be overridden via environment)
export HTTPSERVER_WORKERS="${HTTPSERVER_WORKERS:-2}"
export ENCODE_PROCESS_NUM="${ENCODE_PROCESS_NUM:-1}"
export DECODE_PROCESS_NUM="${DECODE_PROCESS_NUM:-2}"
export ENCODE_PARAL_NUM="${ENCODE_PARAL_NUM:-32}"
export GPT_PARAL_NUM="${GPT_PARAL_NUM:-32}"
export DECODE_PARAL_NUM="${DECODE_PARAL_NUM:-2}"
export DECODE_MAX_BATCH_SIZE="${DECODE_MAX_BATCH_SIZE:-2}"
export MAX_TOTAL_TOKEN_NUM="${MAX_TOTAL_TOKEN_NUM:-32768}"
export MAX_REQ_TOTAL_LEN="${MAX_REQ_TOTAL_LEN:-8192}"
export BATCH_MAX_TOKENS="${BATCH_MAX_TOKENS:-4096}"
export GRAPH_MAX_BATCH_SIZE="${GRAPH_MAX_BATCH_SIZE:-4}"
export DATA_TYPE="${DATA_TYPE:-bfloat16}"
export LOAD_TRT="${LOAD_TRT:-True}"
export DISABLE_CUDAGRAPH="${DISABLE_CUDAGRAPH:-False}"
export MODE="${MODE:-triton_flashdecoding}"

echo "Configuration:"
echo "  Model: $MODEL_DIR"
echo "  Precision: $DATA_TYPE"
echo "  HTTP Workers: $HTTPSERVER_WORKERS"
echo "  Decode Processes: $DECODE_PROCESS_NUM"
echo "  Max Batch Size: $DECODE_MAX_BATCH_SIZE"
echo "  Total Tokens: $MAX_TOTAL_TOKEN_NUM"
echo ""

# Update wsgi_wrapper.py with current config
python -c "
import sys
sys.path.insert(0, 'cosyvoice')

# Read current config
with open('wsgi_wrapper.py', 'r') as f:
    content = f.read()

# Update CONFIG dictionary
import re
config_updates = {
    'model_dir': '$MODEL_DIR',
    'httpserver_workers': $HTTPSERVER_WORKERS,
    'encode_process_num': $ENCODE_PROCESS_NUM,
    'decode_process_num': $DECODE_PROCESS_NUM,
    'encode_paral_num': $ENCODE_PARAL_NUM,
    'gpt_paral_num': $GPT_PARAL_NUM,
    'decode_paral_num': $DECODE_PARAL_NUM,
    'decode_max_batch_size': $DECODE_MAX_BATCH_SIZE,
    'max_total_token_num': $MAX_TOTAL_TOKEN_NUM,
    'max_req_total_len': $MAX_REQ_TOTAL_LEN,
    'batch_max_tokens': $BATCH_MAX_TOKENS,
    'graph_max_batch_size': $GRAPH_MAX_BATCH_SIZE,
    'data_type': '$DATA_TYPE',
    'load_trr': $LOAD_TRT,
    'disable_cudagraph': $DISABLE_CUDAGRAPH,
    'mode': ['$MODE']
}

# Apply updates
for key, value in config_updates.items():
    if isinstance(value, str) and not value.startswith('['):
        value = f'\"{value}\"'
    pattern = f'\"{key}\": [^,}]+'
    replacement = f'\"{key}\": {value}'
    content = re.sub(pattern, replacement, content)

# Write back
with open('wsgi_wrapper.py', 'w') as f:
    f.write(content)
"

# Start the server directly using wsgi_wrapper.py
python wsgi_wrapper.py