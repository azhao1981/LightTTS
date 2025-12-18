#!/bin/bash
# =============================================================================
# LightTTS + CosyVoice2 Deployment Script for NVIDIA A10 (24GB VRAM)
# 基于深度代码审计的生产级部署方案 (可直接执行版本)
#
# Author: Chief AI Architect
# GPU: NVIDIA A10 (24GB VRAM)
# Model: CosyVoice2-0.5B-finetune-v1
# 环境管理: UV + Rye/Venv (自动激活)
# =============================================================================

set -e  # 遇到错误立即退出

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 环境初始化 (自动处理)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 获取绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"  # 光标已在~/tts/light-tts

echo "======================================================================"
echo "LightTTS-A10 环境初始化"
echo "======================================================================"
echo "工作目录: $PROJECT_ROOT"

# 方案1: 检查 .envrc (在 ~/tts 级别)
PARENT_DIR="$(dirname "$PROJECT_ROOT")"  # ~/tts
if [ -f "$PARENT_DIR/.envrc" ]; then
    echo "✓ 加载 .envrc: $PARENT_DIR/.envrc"
    source "$PARENT_DIR/.envrc"
fi

# 方案2: 检查 uv venv (在子目录)
if [ -d "$PROJECT_ROOT/.venv/bin" ]; then
    echo "✓ 激活本地 .venv: $PROJECT_ROOT/.venv"
    source "$PROJECT_ROOT/.venv/bin/activate"
elif [ -d "$PARENT_DIR/tts-research/.venv/bin" ]; then
    echo "✓ 激活 tts-research .venv: $PARENT_DIR/tts-research/.venv"
    source "$PARENT_DIR/tts-research/.venv/bin/activate"
fi

# 验证Python环境
PYTHON_EXE=$(which python)
if [ -z "$PYTHON_EXE" ]; then
    echo "✗ 错误: 未找到Python解释器"
    exit 1
fi

# 关键: 设置PYTHONPATH以导入light_tts模块
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH="$PROJECT_ROOT"
else
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
fi

echo "Python: $PYTHON_EXE"
echo "版本: $(python --version 2>&1)"
echo "PYTHONPATH: $PYTHONPATH"
echo "======================================================================"
echo ""

# 验证模块是否可导入
python -c "import light_tts" 2>/dev/null || {
    echo "✗ 无法导入 light_tts 模块"
    echo "请检查 PYTHONPATH 设置"
    echo "需要将 $PROJECT_ROOT 加入 Python 路径"
    exit 1
}
echo "✓ light_tts 模块加载正常"
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 配置参数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

readonly MODEL_DIR="/root/.cache/modelscope/hub/models/azhao2050/CosyVoice2-0___5B-finetune-v1"
readonly BASE_PORT=8080
readonly GPU_ID=0
readonly CUDA_VISIBLE_DEVICES="0"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 功能函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

check_gpu() {
    log "检查GPU状态..."
    command -v nvidia-smi >/dev/null 2>&1 || { log "警告: 未安装nvidia-smi"; return; }

    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits
    local gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)

    if [ "$gpu_memory" -lt 20000 ]; then
        log "警告: GPU显存小于20GB，建议减少并发"
    fi
}

check_model() {
    if [ ! -d "$MODEL_DIR" ]; then
        log "错误: 模型目录不存在: $MODEL_DIR"
        exit 1
    fi

    # 检查核心模型文件 (LightTTS专用)
    local critical_files=("llm.pt" "flow.pt" "hift.pt" "cosyvoice2.yaml")
    for file in "${critical_files[@]}"; do
        if [ ! -f "$MODEL_DIR/$file" ]; then
            log "错误: 模型文件缺失: $MODEL_DIR/$file"
            exit 1
        fi
    done

    # 检查子目录配置
    if [ ! -d "$MODEL_DIR/CosyVoice-BlankEN" ]; then
        log "错误: CosyVoice-BlankEN 目录缺失"
        exit 1
    fi

    if [ ! -f "$MODEL_DIR/CosyVoice-BlankEN/config.json" ]; then
        log "错误: LLM配置文件缺失"
        exit 1
    fi

    log "模型完整性检查通过 ✓"
}

wait_for_health() {
    local port=$1
    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -s "http://localhost:$port/healthz" > /dev/null 2>&1; then
            log "端口 $port 健康检查通过 ✓"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    log "警告: 端口 $port 启动超时"
    return 1
}

cleanup() {
    log "清理残留进程..."
    pkill -f "python.*api_server" 2>/dev/null
    pkill -f "gicorn.*light_tts.server.api_http" 2>/dev/null
    sleep 2
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 启动模式
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

start_single_worker() {
    local port=$1
    local log_file="/tmp/lighttts_single_${port}.log"

    log "启动单Worker模式 [Port: $port]..."
    log "日志文件: $log_file"

    export CUDA_VISIBLE_DEVICES=$GPU_ID
    export LIGHTLLM_DEBUG=0

    echo "启动命令: python -m light_tts.server.api_server --model_dir $MODEL_DIR --port $port ..."

    # 前台启动并输出到日志
    python -m light_tts.server.api_server \
        --model_dir "$MODEL_DIR" \
        --host "0.0.0.0" \
        --port "$port" \
        --encode_process_num 1 \
        --decode_process_num 1 \
        --max_total_token_num 65536 \
        --max_req_total_len 32768 \
        --batch_max_tokens 16384 \
        --load_trt True \
        --cache_capacity 200 \
        --health_monitor > "$log_file" 2>&1 &

    local pid=$!
    echo $pid > "/tmp/lighttts_worker_single.pid"

    # 等待并监控启动
    sleep 5
    tail -n 20 "$log_file"

    wait_for_health $port
    log "单Worker模式运行中，PID: $pid"
}

start_multi_workers() {
    local port1=$BASE_PORT
    local port2=$((BASE_PORT + 1))

    log "启动双实例模式 (A10负载分摊)..."
    log "实例1: http://localhost:$port1"
    log "实例2: http://localhost:$port2"

    # Worker 1
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    python -m light_tts.server.api_server \
        --model_dir "$MODEL_DIR" \
        --host "0.0.0.0" \
        --port "$port1" \
        --encode_process_num 1 \
        --decode_process_num 1 \
        --max_total_token_num 40960 \
        --max_req_total_len 20480 \
        --batch_max_tokens 10240 \
        --load_trt True \
        --cache_capacity 150 &

    echo $! > "/tmp/lighttts_worker_1.pid"
    log "Worker 1 启动中 (PID: $(cat /tmp/lighttts_worker_1.pid))"
    sleep 20  # 让第一个实例先加载模型

    # Worker 2 (复用同一GPU，降低token缓存)
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    python -m light_tts.server.api_server \
        --model_dir "$MODEL_DIR" \
        --host "0.0.0.0" \
        --port "$port2" \
        --encode_process_num 1 \
        --decode_process_num 1 \
        --max_total_token_num 40960 \
        --max_req_total_len 20480 \
        --batch_max_tokens 10240 \
        --load_trt True \
        --cache_capacity 150 &

    echo $! > "/tmp/lighttts_worker_2.pid"
    log "Worker 2 启动中 (PID: $(cat /tmp/lighttts_worker_2.pid))"

    # 并行检查健康状态
    wait_for_health $port1
    wait_for_health $port2

    log "双实例模式部署完成"
    log "负载均衡建议: nginx/haproxy 分发到 $port1 和 $port2"
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 主流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

main() {
    local mode=${1:-"single"}

    echo ""
    log "======================================================================"
    log "LightTTS-A10 生产部署 (模式: $mode)"
    log "======================================================================"

    # 执行前置检查
    check_gpu
    check_model
    cleanup

    # 启动
    case "$mode" in
        "single")
            start_single_worker $BASE_PORT
            ;;
        "multi")
            start_multi_workers
            ;;
        *)
            echo "用法: $0 [single|multi]"
            echo "  single - 单实例模式 (适合调试，端口8080)"
            echo "  multi  - 双实例模式 (A10负载分摊，端口8080+8081)"
            exit 1
            ;;
    esac

    log "======================================================================"
    log "部署完成"
    log "日志查看: tail -f /tmp/lighttts_*.log"
    log "健康检查: curl http://localhost:$BASE_PORT/healthz"
    log "======================================================================"
}

# 运行 (自动处理环境)
main "$@"
