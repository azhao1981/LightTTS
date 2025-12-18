#!/usr/bin/env python3
"""
A10 Optimized LightTTS Launcher
Optimized for 24GB VRAM with balanced TTFA and throughput
"""

import os
import sys
import argparse

# Add cosyvoice to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cosyvoice'))

def main():
    parser = argparse.ArgumentParser(description='LightTTS A10 Optimized Launcher')
    parser.add_argument('--workers', type=int, default=2, help='Number of HTTP workers')
    parser.add_argument('--decode-proc', type=int, default=2, help='Number of decode processes')
    parser.add_argument('--precision', type=str, default='bfloat16',
                        choices=['float16', 'bfloat16', 'float32'],
                        help='Model precision (bfloat16 recommended for A10)')
    parser.add_argument('--model-dir', type=str,
                        default='./pretrained_models/CosyVoice2-0.5B',
                        help='Model directory')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=8080, help='Port number')

    args = parser.parse_args()

    # Print configuration
    print("=" * 60)
    print("LightTTS A10 Optimized Configuration")
    print("=" * 60)
    print(f"HTTP Workers: {args.workers}")
    print(f"Decode Processes: {args.decode_proc}")
    print(f"Precision: {args.precision}")
    print(f"Model: {args.model_dir}")
    print(f"Address: {args.host}:{args.port}")
    print("=" * 60)

    # Prepare sys.argv for the server
    sys.argv = [
        'api_server.py',
        '--host', args.host,
        '--port', str(args.port),
        '--model_dir', args.model_dir,
        '--httpserver_workers', str(args.workers),
        '--encode_process_num', '1',
        '--decode_process_num', str(args.decode_proc),
        '--encode_paral_num', '32',
        '--gpt_paral_num', '32',
        '--decode_paral_num', '2',
        '--decode_max_batch_size', '2',
        '--max_total_token_num', '32768',
        '--max_req_total_len', '8192',
        '--batch_max_tokens', '4096',
        '--graph_max_batch_size', '4',
        '--data_type', args.precision,
        '--load_trt', 'True',
        '--disable_cudagraph', 'False',
        '--mode', 'triton_flashdecoding',
        '--zmq_mode', 'ipc:///tmp/',
        '--tokenizer_mode', 'slow',
        '--load_way', 'HF',
        '--running_max_req_size', '100',
        '--trust_remote_code', 'False',
        '--disable_log_stats', 'False',
        '--log_stats_interval', '10',
        '--router_token_ratio', '0.0',
        '--router_max_new_token_len', '1024',
        '--router_max_wait_tokens', '8',
        '--cache_capacity', '200',
        '--cache_reserved_ratio', '0.5',
        '--sample_close', 'False',
        '--health_monitor', 'False',
        '--gpt_paral_step_num', '200',
        '--load_jit', 'False',
    ]

    # Import and run the server
    from light_tts.server.api_server import main
    main()

if __name__ == "__main__":
    main()