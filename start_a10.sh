#!/bin/bash

# LightTTS A10 Optimized Startup Script
# Load virtual environment first
cd ~/tts/
source .envrc

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)/cosyvoice"

# Go to light-tts directory
cd light-tts

echo "========================================"
echo "Starting LightTTS with A10 optimizations"
echo "========================================"

# Run with optimized configuration
python run_a10_optimized.py "$@"