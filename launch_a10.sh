#!/bin/bash

# LightTTS A10 Optimized Launcher
# This script sets up environment and launches the server with A10 optimizations

# Load virtual environment
cd ~/tts/
source .envrc

# Set GPU and path
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)/cosyvoice"

# Go to light-tts directory
cd light-tts

echo "========================================"
echo "LightTTS A10 Optimized Launcher"
echo "========================================"

# Parse command line arguments
WORKERS=${1:-2}
DECODE_PROC=${2:-2}
PRECISION=${3:-"bfloat16"}

echo "Configuration:"
echo "  HTTP Workers: $WORKERS"
echo "  Decode Processes: $DECODE_PROC"
echo "  Precision: $PRECISION"
echo "========================================"

# Run the server directly with all parameters
exec python -m light_tts.server.api_server \
    --model_dir ./pretrained_models/CosyVoice2-0.5B \
    --host 0.0.0.0 \
    --port 8080 \
    --httpserver_workers $WORKERS \
    --encode_process_num 1 \
    --decode_process_num $DECODE_PROC \
    --encode_paral_num 32 \
    --gpt_paral_num 32 \
    --decode_paral_num 2 \
    --decode_max_batch_size 2 \
    --max_total_token_num 32768 \
    --max_req_total_len 8192 \
    --batch_max_tokens 4096 \
    --graph_max_batch_size 4 \
    --data_type $PRECISION \
    --load_trt True \
    --disable_cudagraph False \
    --mode triton_flashdecoding \
    --zmq_mode ipc:///tmp/ \
    --tokenizer_mode slow \
    --load_way HF \
    --running_max_req_size 100 \
    --trust_remote_code False \
    --disable_log_stats False \
    --log_stats_interval 10 \
    --router_token_ratio 0.0 \
    --router_max_new_token_len 1024 \
    --router_max_wait_tokens 8 \
    --cache_capacity 200 \
    --cache_reserved_ratio 0.5 \
    --sample_close False \
    --health_monitor False \
    --gpt_paral_step_num 200 \
    --load_jit False