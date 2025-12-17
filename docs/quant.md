1. 量化配置方式
项目支持两种INT8量化方式： 方式一：KV Cache INT8量化（推荐）

# 只量化KV cache，权重保持FP16
python -m light_tts.server.api_server \
  --model_dir ./pretrained_models/CosyVoice2-0.5B-latest \
  --load_trt True \
  --quant_kv_cache int8
方式二：权重INT8量化

# 使用Triton INT8权重量化
python -m light_tts.server.api_server \
  --model_dir ./pretrained_models/CosyVoice2-0.5B-latest \
  --load_trt True \
  --quantize triton_int8weight
2. 量化选项组合
支持的量化模式（来自light_tts/server/api_cli.py）：
triton_int8weight: Triton实现的INT8权重量化
triton_int4weight: INT4权重量化
ppl_int4weight: PPL框架INT4量化
triton-fp8w8a8-b128: FP8精度格式
配合--quant_kv_cache int8进行KV cache量化

3. 建议的渐进式量化策略
第一步: 只量化KV cache (--quant_kv_cache int8)
第二步: 如需更高压缩，使用INT8权重量化 (--quantize triton_int8weight)
第三步: 极致压缩使用INT4权重 (--quantize triton_int4weight)
4. 生产部署配置

```bash
# 方案一：仅KV Cache INT8量化（推荐入门） 失败 disable_cudagraph 也不行
python -m light_tts.server.api_server \
    --host 0.0.0.0 \
    --port 8081 \
    --model_dir ./pretrained_models/CosyVoice2-0.5B-finetune-v1 \
    --load_trt True \
    --max_total_token_num 65536 \
    --max_req_total_len 32768 \
    --mode triton_int8kv \
    --disable_cudagraph

python -m light_tts.server.api_server \
    --host 0.0.0.0 \
    --port 8081 \
    --model_dir ./pretrained_models/CosyVoice2-0.5B-finetune-v1 \
    --load_trt True \
    --max_total_token_num 65536 \
    --max_req_total_len 32768 \
    --mode ppl_int8kv

# 方案二：KV Cache + 权重INT8量化

python -m light_tts.server.api_server \
    --host 0.0.0.0 \
    --port 8080 \
    --model_dir ./pretrained_models/CosyVoice2-0.5B-finetune-v1 \
    --load_trt True \
    --max_total_token_num 65536 \
    --max_req_total_len 32768 \
    --mode triton_int8kv triton_int8weight
    
# 权重INT8 only 这个可以
python -m light_tts.server.api_server \
    --host 0.0.0.0 \
    --port 8080 \
    --model_dir ./pretrained_models/CosyVoice2-0.5B-finetune-v1 \
    --load_trt True \
    --max_total_token_num 65536 \
    --max_req_total_len 32768 \
    --mode triton_int8weight

```
python -m light_tts.server.api_server \
    --host 0.0.0.0 \
    --port 8080 \
    --model_dir ./pretrained_models/CosyVoice2-0.5B-finetune-v1 \
    --load_trt True \
    --max_total_token_num 65536 \
    --max_req_total_len 32768 \
    --mode triton_int8kv triton_int8weight