# 性能优化

1. 性能优化详细说明

- 量化支持: INT8/INT4权重量化、KV Cache量化、FP8精度，支持混合量化策略
- 批处理优化: 动态批处理、Token级批处理、连续批处理等机制
- 缓存机制: 共享内存管理、GPU张量缓存、LRU策略
- 硬件加速: TensorRT集成、自定义CUDA内核、CUDA Graph优化

2. 生产部署方式

- Gunicorn配置: run.sh 提供了完整的生产环境部署脚本
- WSGI包装: wsgi_wrapper.py 处理Gunicorn启动参数
- 多Worker支持: 24GB GPU可支持3个Worker进程处理0.5B模型
- 安全考虑: 避免preload防止CUDA fork问题

3. Encode-LLM-Decode流水线架构

- 微服务设计: 三个独立模块通过ZMQ管道通信
- 数据流: HTTP请求→编码→LLM推理→解码→响应
- 扩展性: 每个阶段支持独立水平扩展
- 通信机制: 共享内存零拷贝 + ZMQ消息队列

## API服务启动

1. FastAPI app 对象位置

  位置: light_tts/server/api_http.py 第 107 行
  app = FastAPI()
  这是一个全局变量，可以通过 light_tts.server.api_http:app 访问。

1. 参数解析方式

  使用 argparse：
  - api_cli.py 的 make_argument_parser() 创建解析器
  - 参数通过 set_env_start_args(args) 存入环境变量
  - FastAPI 启动时通过 get_env_start_args() 从环境变量读取

1. 关键架构发现

  重要：这不是一个简单的 FastAPI 服务！整体架构是：

  api_server.py (主控进程)
      ├── 启动 encode 子进程 (GPU)
      ├── 启动 LLM 子进程 (GPU)  
      ├── 启动 decode 子进程 (GPU)
      └── subprocess 启动 gunicorn (HTTP server)
            └── 加载 api_http:app

  系统已经内置了 Gunicorn 支持（见 api_start.py:179-197），HTTP server 只是轻量级前端，真正的 GPU 计算在子进程中。

  