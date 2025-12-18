import os
import sys
import argparse

# Add cosyvoice to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cosyvoice'))

# Parse command line args and set defaults
def parse_args():
    parser = argparse.ArgumentParser()

    # Basic server config
    parser.add_argument("--host", type=str, default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=os.getenv("PORT", 8080))
    parser.add_argument("--model_dir", type=str, default=os.getenv("MODEL_DIR", "./pretrained_models/CosyVoice2-0.5B"))

    # A10 optimized defaults
    parser.add_argument("--httpserver_workers", type=int, default=int(os.getenv("HTTPSERVER_WORKERS", "2")))
    parser.add_argument("--encode_process_num", type=int, default=int(os.getenv("ENCODE_PROCESS_NUM", "1")))
    parser.add_argument("--decode_process_num", type=int, default=int(os.getenv("DECODE_PROCESS_NUM", "2")))
    parser.add_argument("--encode_paral_num", type=int, default=int(os.getenv("ENCODE_PARAL_NUM", "32")))
    parser.add_argument("--gpt_paral_num", type=int, default=int(os.getenv("GPT_PARAL_NUM", "32")))
    parser.add_argument("--decode_paral_num", type=int, default=int(os.getenv("DECODE_PARAL_NUM", "2")))
    parser.add_argument("--decode_max_batch_size", type=int, default=int(os.getenv("DECODE_MAX_BATCH_SIZE", "2")))

    # Memory optimization
    parser.add_argument("--max_total_token_num", type=int, default=int(os.getenv("MAX_TOTAL_TOKEN_NUM", "32768")))
    parser.add_argument("--max_req_total_len", type=int, default=int(os.getenv("MAX_REQ_TOTAL_LEN", "8192")))
    parser.add_argument("--batch_max_tokens", type=int, default=int(os.getenv("BATCH_MAX_TOKENS", "4096")))
    parser.add_argument("--graph_max_batch_size", type=int, default=int(os.getenv("GRAPH_MAX_BATCH_SIZE", "4")))

    # Precision and optimization
    parser.add_argument("--data_type", type=str, default=os.getenv("DATA_TYPE", "bfloat16"),
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--load_trt", type=bool, default=os.getenv("LOAD_TRT", "True").lower() == "true")
    parser.add_argument("--disable_cudagraph", type=bool, default=os.getenv("DISABLE_CUDAGRAPH", "False").lower() == "true")
    parser.add_argument("--mode", type=str, default=os.getenv("MODE", "triton_flashdecoding"), nargs='+')

    return parser.parse_args()

args = parse_args()

# Modify sys.argv for the actual server
sys.argv = [
    "api_server.py",
    "--host", str(args.host),
    "--port", str(args.port),
    "--model_dir", args.model_dir,
    "--httpserver_workers", str(args.httpserver_workers),
    "--encode_process_num", str(args.encode_process_num),
    "--decode_process_num", str(args.decode_process_num),
    "--encode_paral_num", str(args.encode_paral_num),
    "--gpt_paral_num", str(args.gpt_paral_num),
    "--decode_paral_num", str(args.decode_paral_num),
    "--decode_max_batch_size", str(args.decode_max_batch_size),
    "--max_total_token_num", str(args.max_total_token_num),
    "--max_req_total_len", str(args.max_req_total_len),
    "--batch_max_tokens", str(args.batch_max_tokens),
    "--graph_max_batch_size", str(args.graph_max_batch_size),
    "--data_type", args.data_type,
    "--load_trt", str(args.load_trt),
    "--disable_cudagraph", str(args.disable_cudagraph),
    "--mode"
] + args.mode

# Import and run the server
from light_tts.server.api_server import main
main()