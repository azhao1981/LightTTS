import sys
import os

sys.argv = [
    "api_server", 
    "--model_dir", "./pretrained_models/CosyVoice2-0.5B-finetune-v1",
    "--load_trt", "True",
    "--max_total_token_num", "131072",
    "--max_req_total_len", "32768",
    "--host", "0.0.0.0",
    "--port", "8080"
]

try:
    from light_tts.server.api_http import app
except Exception as e:
    print(f"❌ Failed to initialize LightTTS APIServer: {e}")
    raise