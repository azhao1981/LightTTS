python -m light_tts.server.api_server \
  --httpserver_workers 3 \
  --encode_process_num 2 \
  --decode_process_num 2 \
  --gpt_paral_num 100 \
  --load_trt True \
  --max_total_token_num 65536 \
  --max_req_total_len 32768 \
  --model_dir ./pretrained_models/CosyVoice2-0.5B-finetune-v1

# python -m light_tts.server.api_server \
#   --model_dir ./pretrained_models/CosyVoice2-0.5B-latest \
#   --load_trt True \
#   --max_total_token_num 65536 \
#   --max_req_total_len 32768
