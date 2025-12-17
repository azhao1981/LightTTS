1. 启动命令

  直接运行：
  python wsgi_wrapper.py

  这会：
  1. 启动 encode/LLM/decode 子进程
  2. 设置环境变量
  3. 启动 Gunicorn (3 workers)

  2. 配置修改

  在 wsgi_wrapper.py 顶部的 CONFIG 字典中修改参数：
  CONFIG = {
      "model_dir": "./pretrained_models/CosyVoice2-0.5B-finetune-v1",
      "load_trt": True,
      "max_total_token_num": 131072,
      "max_req_total_len": 32768,
      "host": "0.0.0.0",
      "port": 8080,
      "httpserver_workers": 3,  # Gunicorn workers 数量
      ...
  }

  ---
  重要提醒：关于 GPU 利用率

  增加 HTTP workers 不会提高 GPU 利用率！

  架构分析：
  HTTP Workers (CPU)  ──▶  encode 进程 (GPU)
                           ──▶  LLM 进程 (GPU)
                                ──▶  decode 进程 (GPU)

  - HTTP workers 只处理 HTTP 请求（CPU 密集）
  - GPU 计算在 encode/LLM/decode 进程中
  - 单实例 7GB 显存主要被这些子进程占用

  要提高 GPU 利用率，你应该调整这些参数：

  | 参数                  | 说明              | 建议值   |
  |-----------------------|-------------------|----------|
  | decode_process_num    | decode 进程数     | 尝试 2-3 |
  | decode_paral_num      | decode 并行度     | 尝试 2-4 |
  | decode_max_batch_size | decode batch 大小 | 尝试 4-8 |
  | gpt_paral_num         | GPT 并行度        | 保持 50  |

  真正榨干 24GB 显存的方法：

  CONFIG = {
      ...
      "decode_process_num": 2,      # 增加 decode 进程
      "decode_paral_num": 2,        # 增加 decode 并行
      "decode_max_batch_size": 4,   # 增加 batch size
      "httpserver_workers": 3,
      ...
  }

  或者启动多个完整服务实例（不同端口），用 nginx 做负载均衡。