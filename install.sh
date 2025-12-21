sudo apt-get update 

sudo apt-get install -y \
    build-essential \
    python3-dev \
    git \
    wget \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    libasound2-dev

UV_TORCH_BACKEND=cu121 uv pip install torch==2.3.1 torchaudio==2.3.1
uv pip install pynini==2.1.5
uv pip install -r requirements_lighttts.txt \
  -i https://mirrors.aliyun.com/pypi/simple \
  -c requirements_lighttts_constraints.txt

sudo apt-get install sox libsox-dev -y

python down.py

# onnxruntime
uv pip install onnxruntime-gpu==1.18.0 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
DS_SKIP_CUDA_CHECK=1 uv pip install deepspeed==0.15.1 -i "https://mirrors.aliyun.com/pypi/simple"

# tensorrt
# 需要用 python -m 安装
python -m pip install tensorrt-cu12==10.0.1 tensorrt-cu12-bindings==10.0.1 tensorrt-cu12-libs==10.0.1
python -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"

# ttsfrd
cd pretrained_models/CosyVoice-ttsfrd/
sudo apt install unzip -y
unzip resource.zip -d .
uv pip install ttsfrd_dependency-0.1-py3-none-any.whl
uv pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
python -c "import ttsfrd; print('✅ ttsfrd installed successfully')"


source .envrc && python -m light_tts.server.api_server \
    --host 0.0.0.0 \
    --port 8080 \
    --model_dir ./pretrained_models/CosyVoice2-0.5B-finetune-v1 \
    --load_trt True \
    --data_type bfloat16 \
    --flow_steps 5 \
    --max_total_token_num 65536 \
    --max_req_total_len 32768

# 这个需要安装
uv pip install triton==3.1.0 --no-deps