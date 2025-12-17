import sys
import os

# -------------------------------------------------------------------
# 1.在这里配置你的启动参数
# Gunicorn 启动时没法传参给 App，所以我们在这里“硬编码”或者读环境变量
# -------------------------------------------------------------------
# 模拟命令行参数，让 LightTTS 以为自己是从命令行启动的
sys.argv = [
    "api_server",  # 程序名，随便填
    "--model_dir", "./pretrained_models/CosyVoice2-0.5B-finetune-v1",
    "--load_trt", "True",
    "--max_total_token_num", "131072",  # 显存够大，我把这个翻倍了以支持更高并发
    "--max_req_total_len", "32768",
    "--host", "0.0.0.0",  # 这些其实 Gunicorn 会接管，但填上也无妨
    "--port", "8080"
]

# 2. 导入 LightTTS 的 app 对象
# 注意：这一步会触发 LightTTS 的初始化逻辑
# 只有在 Worker 进程启动后才会执行到这里，所以 CUDA 初始化是安全的（每个 Worker 独立）
try:
    from light_tts.server.api_server import app
except ImportError:
    print("❌ 找不到 light_tts 模块，请确认你在正确的目录下，或者已安装 light_tts")
    raise