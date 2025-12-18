#!/bin/bash
# =============================================================================
# LightTTS GPU 显存监控脚本
# 用于实时监控CosyVoice2模型在A10上的显存占用
# =============================================================================

INTERVAL=${1:-2}  # 默认2秒刷新
echo "监控GPU显存使用 (间隔: ${INTERVAL}s)"
echo "按 Ctrl+C 退出"
echo "======================================================================"

while true; do
    clear
    echo "GPU实时监控 - $(date +'%Y-%m-%d %H:%M:%S')"
    echo "======================================================================"

    # 显存使用
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
    awk -F', ' '
    {
        printf "GPU %d: %s\n", $1, $2
        used = $3
        total = $4
        util = $5
        pct = (used/total)*100
        printf "  显存: %d / %d MB (%.1f%%)\n", used, total, pct
        printf "  利用率: %d%%\n", util
        printf "  进程: "
        system("nvidia-smi pmon -c 1 | grep \"^( "$1\" )\" | awk \047{print \"PID \" $2 \" - \" $4 \"\\n\"}\047")
    }'

    # 显示LightTTS相关进程
    echo ""
    echo "LightTTS进程:"
    ps aux | grep -E "python.*(api_server|tts_encode|tts_llm|tts_decode)" --color=auto | grep -v grep || echo "  无活跃进程"

    echo ""
    echo "======================================================================"
    echo "下一次更新在 ${INTERVAL} 秒后..."
    sleep $INTERVAL
done
