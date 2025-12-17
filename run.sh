
python -m light_tts.server.api_server \
  --host 0.0.0.0 \
  --port 8080 \
  --httpserver_workers 6 \
  --encode_process_num 4 \
  --decode_process_num 3 \
  --gpt_paral_num 200 \
  --encode_paral_num 100 \
  --decode_paral_num 10 \
  --load_trt True \
  --max_total_token_num 131072 \
  --max_req_total_len 32768 \
  --model_dir ./pretrained_models/CosyVoice2-0.5B-finetune-v1
# HTTP Workers: 6个并发HTTP处理
# Encode Processes: 4个特征提取进程
# LLM Processes: 3个独立的LLM推理进程（每进程约6GB显存）
# Decode Processes: 3个TensorRT解码进程
# 总显存: 3 × 6GB = 18GB + 系统开销 ≈ 20GB

# 这个并不能利用显存 
# python -m light_tts.server.api_server \
#   --httpserver_workers 3 \
#   --host 0.0.0.0 \
#   --port 8080 \
#   --encode_process_num 2 \
#   --decode_process_num 1 \
#   --gpt_paral_num 100 \
#   --load_trt True \
#   --max_total_token_num 65536 \
#   --max_req_total_len 32768 \
#   --model_dir ./pretrained_models/CosyVoice2-0.5B-finetune-v1

# 官方启动
# python -m light_tts.server.api_server \
#   --model_dir ./pretrained_models/CosyVoice2-0.5B-latest \
#   --load_trt True \
#   --max_total_token_num 65536 \
#   --max_req_total_len 32768
