#!/bin/bash

PORT=8070
CUDA_VISIBLE_DEVICES=0

pkill -f "port $PORT" || true
sleep 2

source /usr/local/tts/.venv/bin/activate
cd /usr/local/tts/LightTTS.dev

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES LIGHTTTS_STREAM_AUDIO_MAX_SIZE=524288 exec python -m light_tts.server.api_server \
    --host 0.0.0.0 \
    --port $PORT \
    --model_dir ./pretrained_models/CosyVoice2-0.5B-finetune-v1 \
    --load_trt True \
    --max_total_token_num 120000 \
    --max_req_total_len 4096 \
    --batch_max_tokens 4096 \
    --mode triton_flashdecoding triton_int8weight \
    --disable_log_stats \
    --decode_max_batch_size 1 \
    --decode_paral_num 3 \
    --gpt_paral_num 80 \
    --gpt_paral_step_num 50 \
    --router_max_wait_tokens 2 \
    --cache_capacity 500 > /var/log/tts/cosy01.dev.log 2>&1