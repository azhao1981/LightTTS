"""
WSGI Wrapper for LightTTS with Gunicorn

This wrapper initializes the LightTTS subprocesses and exports the FastAPI app.
Use with launcher.py for proper subprocess management.

Usage:
    python launcher.py

Or directly (not recommended for production):
    python wsgi_wrapper.py --start-subprocesses
    # Then in another terminal:
    gunicorn -w 3 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8080 wsgi_wrapper:app
"""

import sys
import os
import json

# ============================================================================
# Configuration - Modify these parameters as needed
# ============================================================================
CONFIG = {
    "model_dir": "./pretrained_models/CosyVoice2-0.5B",
    "load_trt": True,
    "max_total_token_num": 32768,  # A10 optimized: reduced for 24GB
    "max_req_total_len": 8192,
    "host": "0.0.0.0",
    "port": 8080,
    "httpserver_workers": 2,  # A10 optimized: 2 workers for 24GB
    # Subprocess configuration
    "encode_process_num": 1,
    "decode_process_num": 2,  # A10 optimized: 2 decode processes
    "encode_paral_num": 32,  # A10 optimized: reduced to prevent OOM
    "gpt_paral_num": 32,
    "decode_paral_num": 2,  # Allow 2 concurrent decodes
    "decode_max_batch_size": 2,  # A10 optimized: batch size 2
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
    "disable_cudagraph": False,
    "graph_max_batch_size": 16,
    "graph_max_len_in_batch": 32768,
    "load_jit": False,
    "batch_max_tokens": 4096,  # Control prefill batch size
    "graph_max_batch_size": 4,  # CUDA Graph optimization
    "mode": ["triton_flashdecoding"],  # Flash Attention for A10
    "data_type": "bfloat16",  # A10 optimized precision
}


def init_subprocesses():
    """
    Initialize all LightTTS subprocesses (encode, LLM, decode).
    This should only be called ONCE before starting Gunicorn workers.
    """
    import torch
    torch.multiprocessing.set_start_method("spawn", force=True)

    import multiprocessing as mp
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent / 'cosyvoice'))

    from easydict import EasyDict
    from light_tts.utils.net_utils import alloc_can_use_network_port, PortLocker
    from light_tts.utils.envs_utils import set_env_start_args, set_unique_server_name
    from light_tts.utils.start_utils import process_manager
    from light_tts.server.tts_encode.manager import start_tts1_encode_process
    from light_tts.server.tts_llm.manager import start_tts_llm_process
    from light_tts.server.tts_decode.manager import start_tts_decode_process
    from light_tts.utils.log_utils import init_logger

    logger = init_logger(__name__)

    # Convert config to args object
    args = EasyDict(CONFIG.copy())

    # Validation
    assert args.max_req_total_len <= args.max_total_token_num
    assert args.zmq_mode in ["tcp://", "ipc:///tmp/"]

    # Set unique server name
    set_unique_server_name(args)

    # Adjust zmq_mode for IPC
    if args.zmq_mode == "ipc:///tmp/":
        from light_tts.utils.envs_utils import get_unique_server_name
        zmq_mode = f"{args.zmq_mode}_{get_unique_server_name()}_"
        args.zmq_mode = zmq_mode
        logger.info(f"zmq mode head: {args.zmq_mode}")

    # Set batch_max_tokens
    if args.batch_max_tokens is None:
        batch_max_tokens = int(1 / 6 * args.max_total_token_num)
        batch_max_tokens = max(batch_max_tokens, args.max_req_total_len)
        args.batch_max_tokens = batch_max_tokens

    # Lock port
    ports_locker = PortLocker([args.port])
    ports_locker.lock_port()

    num_loras = max(args.decode_process_num, 1)

    # Allocate ports
    can_use_ports = alloc_can_use_network_port(
        num=num_loras * 2 + args.encode_process_num + 100, used_nccl_port=None
    )

    httpserver_port = can_use_ports[0]
    del can_use_ports[0]
    tts1_encode_ports = can_use_ports[0:args.encode_process_num]
    del can_use_ports[0:args.encode_process_num]

    args.httpserver_port = httpserver_port
    args.tts1_encode_ports = tts1_encode_ports

    tts_llm_ports = can_use_ports[0:num_loras]
    del can_use_ports[0:num_loras]

    # Set data_type for models if specified
    if hasattr(args, 'data_type') and args.data_type:
        os.environ["LIGHTTTS_DATA_TYPE"] = args.data_type

    # Set environment variables for Gunicorn workers
    set_env_start_args(args)
    logger.info(f"all start args: {args}")
    ports_locker.release_port()

    # Start encode processes
    funcs = []
    start_args = []
    encode_parall_lock = mp.Semaphore(args.encode_paral_num)
    for index_id in range(args.encode_process_num):
        funcs.append(start_tts1_encode_process)
        start_args.append((args, tts_llm_ports, tts1_encode_ports[index_id], index_id, encode_parall_lock))
    process_manager.start_submodule_processes(start_funcs=funcs[0:1], start_args=start_args[0:1])

    tts_decode_ports = can_use_ports[0:num_loras]
    del can_use_ports[0:num_loras]

    # Start LLM processes
    funcs = []
    start_args = []
    gpt_parall_lock = mp.Semaphore(args.gpt_paral_num)
    style_names = ["CosyVoice2"] * num_loras
    for style_name, tts_llm_port, tts_decode_port in zip(style_names, tts_llm_ports, tts_decode_ports):
        funcs.append(start_tts_llm_process)
        start_args.append((args, tts_llm_port, tts_decode_port, style_name, gpt_parall_lock))
    process_manager.start_submodule_processes(start_funcs=funcs, start_args=start_args)

    # Start decode processes
    decode_parall_lock = mp.Semaphore(args.decode_paral_num)
    funcs = []
    start_args = []
    for decode_proc_index in range(args.decode_process_num):
        tmp_args = []
        for style_name, tts_decode_port in zip(style_names, tts_decode_ports):
            tmp_args.append((args, tts_decode_port, httpserver_port, style_name, decode_parall_lock, decode_proc_index))
        funcs.append(start_tts_decode_process)
        start_args.append((tmp_args,))
    process_manager.start_submodule_processes(start_funcs=funcs, start_args=start_args)

    logger.info("All subprocesses started successfully!")
    return args


# Check if we should start subprocesses (main process mode)
if __name__ == "__main__" or os.environ.get("LIGHTTTS_INIT_SUBPROCESSES") == "1":
    if os.environ.get("LIGHTTTS_SUBPROCESS_READY") != "1":
        print("=" * 60)
        print("Starting LightTTS subprocesses...")
        print("=" * 60)

        args = init_subprocesses()
        os.environ["LIGHTTTS_SUBPROCESS_READY"] = "1"

        if __name__ == "__main__":
            print("\n" + "=" * 60)
            print("Subprocesses ready! Starting Gunicorn...")
            print("=" * 60)

            import subprocess
            cmd = [
                "gunicorn",
                "-w", str(CONFIG["httpserver_workers"]),
                "-k", "uvicorn.workers.UvicornWorker",
                "-b", f"{CONFIG['host']}:{CONFIG['port']}",
                "--timeout", "300",
                "--access-logfile", "-",
                "--error-logfile", "-",
                "wsgi_wrapper:app"
            ]
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd)
else:
    # Gunicorn worker mode - just import the app
    if os.environ.get("LIGHTLLM_START_ARGS") is None:
        raise RuntimeError(
            "LightTTS environment not initialized!\n"
            "Please run: python wsgi_wrapper.py\n"
            "Or use: python launcher.py"
        )

# Import the FastAPI app
from light_tts.server.api_http import app

__all__ = ["app"]
