import sys
import os
from light_tts.server.api_server import APIServer

# -------------------------------------------------------------------
# 1. Configure startup parameters
# -------------------------------------------------------------------
# We manually set sys.argv so APIServer parses them as if run from CLI
sys.argv = [
    "api_server", 
    "--model_dir", "./pretrained_models/CosyVoice2-0.5B-finetune-v1",
    "--load_trt", "True",
    "--max_total_token_num", "131072",
    "--max_req_total_len", "32768",
    "--host", "0.0.0.0",
    "--port", "8080"
]

# -------------------------------------------------------------------
# 2. Instantiate the Server and expose 'app'
# -------------------------------------------------------------------
try:
    # Initialize the APIServer class
    # This will parse sys.argv and load the model
    server_instance = APIServer()
    
    # Expose the internal FastAPI object for Gunicorn to use
    # Gunicorn looks for a variable named 'app' (or whatever is specified before ':')
    app = server_instance.app 

except Exception as e:
    print(f"❌ Failed to initialize LightTTS APIServer: {e}")
    raise