#!/bin/bash

# Simple A10 Optimized LightTTS Launcher

# Set environment
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)/cosyvoice"

# Parse command line arguments
WORKERS=${1:-2}
DECODE_PROC=${2:-2}
PRECISION=${3:-"bfloat16"}

echo "========================================"
echo "LightTTS A10 Optimized Launcher"
echo "========================================"
echo "HTTP Workers: $WORKERS"
echo "Decode Processes: $DECODE_PROC"
echo "Precision: $PRECISION"
echo "========================================"

# Create a temporary config file with A10 optimizations
cat > wsgi_wrapper_config.py << 'EOF'
import os
import sys

# Add cosyvoice to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cosyvoice'))

# A10 Optimized configuration
CONFIG = {
    "model_dir": os.getenv("MODEL_DIR", "./pretrained_models/CosyVoice2-0.5B"),
    "load_trt": True,
    "max_total_token_num": 32768,  # Optimized for 24GB
    "max_req_total_len": 8192,
    "host": "0.0.0.0",
    "port": 8080,
    "httpserver_workers": int(os.getenv("HTTPSERVER_WORKERS", "2")),
    # A10 optimized subprocess config
    "encode_process_num": 1,
    "decode_process_num": int(os.getenv("DECODE_PROCESS_NUM", "2")),
    "encode_paral_num": 32,  # Reduced to prevent OOM
    "gpt_paral_num": 32,
    "decode_paral_num": 2,  # Allow concurrent decodes
    "decode_max_batch_size": 2,  # Batch size 2 for better utilization
    # Performance optimizations
    "batch_max_tokens": 4096,  # Control prefill batch
    "graph_max_batch_size": 4,  # CUDA Graph optimization
    "mode": ["triton_flashdecoding"],  # Flash Attention
    "data_type": os.getenv("DATA_TYPE", "bfloat16"),
    "disable_cudagraph": False,
    # Other defaults
    "zmq_mode": "ipc:///tmp/",
    "tokenizer_mode": "slow",
    "load_way": "HF",
    "running_max_req_size": 100,
    "trust_remote_code": False,
    "disable_log_stats": False,
    "log_stats_interval": 10,
    "router_token_ratio": 0.0,
    "router_max_new_token_len": 1024,
    "router_max_wait_tokens": 8,
    "cache_capacity": 200,
    "cache_reserved_ratio": 0.5,
    "sample_close": False,
    "health_monitor": False,
    "graph_max_len_in_batch": 32768,
    "load_jit": False,
}

# Set environment variables
os.environ["HTTPSERVER_WORKERS"] = str(CONFIG["httpserver_workers"])
os.environ["DECODE_PROCESS_NUM"] = str(CONFIG["decode_process_num"])
os.environ["DATA_TYPE"] = CONFIG["data_type"]
os.environ["MODEL_DIR"] = CONFIG["model_dir"]

# Import the wsgi_wrapper which will use these settings
exec(open('wsgi_wrapper.py').read())
EOF

# Set environment variables
export HTTPSERVER_WORKERS=$WORKERS
export DECODE_PROCESS_NUM=$DECODE_PROC
export DATA_TYPE=$PRECISION

# Run with our custom config
python wsgi_wrapper_config.py