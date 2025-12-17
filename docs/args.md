python -m light_tts.server.api_server \
  --model_dir ./pretrained_models/CosyVoice2-0.5B-latest \
  --load_trt True \
  --max_total_token_num 65536 \
  --max_req_total_len 32768

python -m light_tts.server.api_server \
    --host 0.0.0.0 \
    --port 8080 \
    --model_dir ./pretrained_models/CosyVoice2-0.5B \
    --load_trt True \
    --max_total_token_num 65536 \
    --max_req_total_len 32768


参数详细说明

并行参数详解
--httpserver_workers 3
类型: int，默认值: 1
作用: HTTP服务器worker进程数量
说明:
控制FastAPI应用的worker进程数
每个worker独立处理HTTP请求
通过Gunicorn启动多个uvicorn worker
3个worker可以并发处理3个HTTP请求
--encode_process_num 2
类型: int，默认值: 1
作用: Encode模块进程数量
说明:
启动2个独立的TTS编码进程
每个进程处理音频特征提取和文本预处理
通过信号量控制并行度(--encode_paral_num)
为每个进程分配独立的端口号
--decode_process_num 1
类型: int，默认值: 1
作用: Decode模块进程数量
说明:
启动1个TTS解码进程
负责将语音token转换为音频波形
目前受num_loras=1限制，最大只能为1
支持TensorRT加速
--gpt_paral_num 100
类型: int，默认值: 50
作用: LLM模块并行度（信号量控制）
说明:
控制LLM推理阶段的最大并发任务数
通过gpt_parall_lock = mp.Semaphore(args.gpt_paral_num)实现
不是进程数，而是单个进程内的并行度
提高到100可以增加LLM推理吞吐量
架构层级关系
HTTP层:     3个HTTP workers (处理3个并发HTTP请求)
    ↓
Encode层:   2个Encode进程 (并行提取特征)
    ↓  
LLM层:      1个LLM进程，内部100个并行任务 (高吞吐量推理)
    ↓
Decode层:   1个Decode进程 (TensorRT加速解码)

--model_dir
类型: str (必需)
默认值: None
作用: 模型权重目录路径
说明: 应用会从该目录加载配置文件、模型权重和分词器
示例: ./pretrained_models/CosyVoice2-0.5B-latest
--load_trt
类型: bool
默认值: True
作用: 是否使用TensorRT加载flow_decoder
说明:
True: 启用TensorRT加速推理，显著提升性能
False: 使用标准PyTorch推理
建议: 生产环境建议开启，需要TensorRT环境支持
--max_total_token_num
类型: int
默认值: 64 * 1024 (65536)
作用: GPU和模型能支持的总token数量
计算公式: max_batch * (input_len + output_len)
说明:
控制并发处理的token容量
设置为65536表示可同时处理约65536个token
影响内存使用和并发能力
--max_req_total_len
类型: int
默认值: 32 * 1024 (32768)
作用: 单个请求的最大长度 (输入+输出)
说明:
限制单个请求的token数量
32768对应CosyVoice2模型的max_position_embeddings
防止单个请求占用过多资源
参数间的关系
内存约束: max_total_token_num 决定总体GPU内存使用
请求限制: max_req_total_len 限制单个请求大小
并发计算: 理论并发数 ≈ max_total_token_num / avg_request_length
推荐配置
24GB GPU + 0.5B模型:
--max_total_token_num 65536    # 总token容量
--max_req_total_len 32768     # 单请求最大长度
更大GPU或模型:
--max_total_token_num 131072  # 翻倍容量
--max_req_total_len 32768     # 保持不变(模型限制)

