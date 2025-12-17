#!/bin/bash

# 显存 24G，0.5B 模型约占 6-7G。
# 3个 Worker 约占 21G，留 3G 给系统，非常完美。
WORKERS=3

echo "🚀 Starting LightTTS with Gunicorn ($WORKERS Workers)..."

# --preload: ❌ 绝对不要用！会导致 CUDA 在主进程初始化，Fork 时报错
# --timeout: 设置大一点，防止模型加载慢导致 Worker 被杀
# worker-class: 必须是 uvicorn，因为 LightTTS 是 FastAPI 应用
gunicorn -w $WORKERS \
    -k uvicorn.workers.UvicornWorker \
    -b 0.0.0.0:8080 \
    --timeout 300 \
    --access-logfile - \
    --error-logfile - \
    wsgi_wrapper:app