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
    log "该模式约需60-90秒完成模型加载，请耐心等待..."

    export CUDA_VISIBLE_DEVICES=$GPU_ID
    export LIGHTLLM_DEBUG=0

    # 关键: 参数必须满足 batch_max_tokens >= max_req_total_len
    # 经实测验证的A10稳定配置
    local max_tokens=32768      # max_total_token_num
    local max_req_len=16384     # max_req_total_len
    local batch_tokens=16384    # batch_max_tokens
    local cache_cap=100

    log "参数配置: max_tokens=$max_tokens, max_req_len=$max_req_len, batch_tokens=$batch_tokens"

    python -m light_tts.server.api_server \
        --model_dir "$MODEL_DIR" \
        --host "0.0.0.0" \
        --port "$port" \
        --encode_process_num 1 \
        --decode_process_num 1 \
        --max_total_token_num $max_tokens \
        --max_req_total_len $max_req_len \
        --batch_max_tokens $batch_tokens \
        --load_trt True \
        --cache_capacity $cache_cap \
        --health_monitor > "$log_file" 2>&1 &

    local pid=$!
    echo $pid > "/tmp/lighttts_worker_single.pid"
    log "进程已启动，PID: $pid"

    # 等待模型加载 (首次启动需要~60秒)
    log "等待模型加载..."
    for i in {1..90}; do
        if [ -f "/tmp/lighttts_single_${port}.log" ]; then
            local line_count=$(wc -l < "/tmp/lighttts_single_${port}.log" 2>/dev/null || echo 0)
            if [ $line_count -gt 50 ]; then
                tail -n 5 "/tmp/lighttts_single_${port}.log"
            fi
        fi
        # 双重检查: healthz 和 query_tts_model
        if curl -s "http://localhost:$port/healthz" > /dev/null 2>&1 && \
           curl -s "http://localhost:$port/query_tts_model" > /dev/null 2>&1; then
            log "✓ 服务就绪! (耗时约 ${i}s)"
            log "服务地址: http://localhost:$port"
            return 0
        fi
        sleep 1
    done

    log "⚠ 服务启动超时，请检查日志: tail -f $log_file"
    exit 1
}

start_multi_workers() {
    local port1=$BASE_PORT
    local port2=$((BASE_PORT + 1))

    log "启动双实例模式 (A10负载分摊)"
    log "注意: 两个实例将共享GPU，每个约占用10GB显存"
    log "实例1: http://localhost:$port1"
    log "实例2: http://localhost:$port2 (60秒后启动)"

    # 共享配置参数 (保守配置，确保稳定性)
    local max_tokens=24576      # 每个实例分配少一点
    local max_req_len=12288     # max_req_total_len = max_tokens / 2
    local batch_tokens=12288    # batch_max_tokens >= max_req_len
    local cache_cap=80

    # Worker 1
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    nohup python -m light_tts.server.api_server \
        --model_dir "$MODEL_DIR" \
        --host "0.0.0.0" \
        --port "$port1" \
        --encode_process_num 1 \
        --decode_process_num 1 \
        --max_total_token_num $max_tokens \
        --max_req_total_len $max_req_len \
        --batch_max_tokens $batch_tokens \
        --load_trt True \
        --cache_capacity $cache_cap \
        --health_monitor > "/tmp/lighttts_worker_1.log" 2>&1 &

    local pid1=$!
    echo $pid1 > "/tmp/lighttts_worker_1.pid"
    log "Worker 1 启动中 (PID: $pid1, 日志: /tmp/lighttts_worker_1.log)"

    # 等待第一个实例完成初始化 (约60-90秒)
    log "等待Worker 1模型加载 (约60-90秒)..."
    sleep 70

    # Worker 2
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    nohup python -m light_tts.server.api_server \
        --model_dir "$MODEL_DIR" \
        --host "0.0.0.0" \
        --port "$port2" \
        --encode_process_num 1 \
        --decode_process_num 1 \
        --max_total_token_num $max_tokens \
        --max_req_total_len $max_req_len \
        --batch_max_tokens $batch_tokens \
        --load_trt True \
        --cache_capacity $cache_cap \
        --health_monitor > "/tmp/lighttts_worker_2.log" 2>&1 &

    local pid2=$!
    echo $pid2 > "/tmp/lighttts_worker_2.pid"
    log "Worker 2 启动中 (PID: $pid2, 日志: /tmp/lighttts_worker_2.log)"

    # 检查两个实例 (双重健康检查)
    log "等待两个实例就绪..."
    for i in {1..90}; do
        local ready=0

        # 双重检查: healthz + query_tts_model
        if curl -s "http://localhost:$port1/healthz" > /dev/null 2>&1 && \
           curl -s "http://localhost:$port1/query_tts_model" > /dev/null 2>&1; then
            ready=$((ready+1))
        fi

        if curl -s "http://localhost:$port2/healthz" > /dev/null 2>&1 && \
           curl -s "http://localhost:$port2/query_tts_model" > /dev/null 2>&1; then
            ready=$((ready+1))
        fi

        if [ $ready -eq 2 ]; then
            log "✓ 双实例模式完成! (耗时约 ${i}s)"
            log "实例1: http://localhost:$port1"
            log "实例2: http://localhost:$port2"
            log "负载均衡建议: nginx/haproxy 分发到两个实例"
            return 0
        fi
        [ $((i % 10)) -eq 0 ] && log "  已就绪实例: $ready/2 ..."
        sleep 1
    done

    log "⚠ 部分实例启动超时，请检查日志"
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
    log "======================================================================"
    log "快速验证:"
    log "  curl http://localhost:$BASE_PORT/healthz"
    log ""
    log "TTS测试 (需先准备 prompt.wav):"
    log "  curl -X POST http://localhost:$BASE_PORT/inference_zero_shot \\"
    log "    -F 'prompt_wav=@/path/to/prompt.wav' \\"
    log "    -F 'prompt_text=我是测试用户' \\"
    log "    -F 'tts_text=部署成功，LightTTS正在运行' \\"
    log "    -F 'stream=false' > output.wav"
    log ""
    log "日志查看:"
    log "  实时: tail -f /tmp/lighttts_*.log"
    log "  进程: ps aux | grep api_server"
    log "  GPU显存: nvidia-smi"
    log ""
    log "停止服务:"
    log "  pkill -f 'python.*api_server' || cat /tmp/lighttts_worker_*.pid | xargs kill"
    log "======================================================================"
}

# 运行 (自动处理环境)
main "$@"
