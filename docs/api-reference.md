# API 参考文档

## 概述

Light TTS 提供 REST API 和 WebSocket API 两种接口，支持零样本语音合成、流式处理和实时双向通信。

## 基础信息

- **基础URL**: `http://localhost:8080`
- **API版本**: v1
- **内容类型**: `multipart/form-data` 或 `application/json`
- **音频格式**: 16kHz WAV

## REST API

### 1. 零样本语音合成

#### 端点
```
POST /inference_zero_shot
```

#### 请求参数

| 参数 | 类型 | 必需 | 默认值 | 描述 |
|------|------|------|--------|------|
| `tts_text` | string | 是 | - | 要合成的文本内容 |
| `prompt_text` | string | 是 | - | 提示文本，描述语音风格 |
| `prompt_wav` | file | 是 | - | 参考音频文件 (WAV格式) |
| `stream` | bool | 否 | false | 是否启用流式返回 |
| `tts_model_name` | string | 否 | "default" | TTS模型名称 |
| `speed` | float | 否 | 1.0 | 语速调节 (0.5-2.0) |

#### 请求示例

**非流式请求**:
```bash
curl -X POST "http://localhost:8080/inference_zero_shot" \
  -F "tts_text=你好，这是一个测试。" \
  -F "prompt_text=这是一个友好的声音" \
  -F "prompt_wav=@reference.wav" \
  -F "stream=false" \
  -F "speed=1.0"
```

**流式请求**:
```bash
curl -X POST "http://localhost:8080/inference_zero_shot" \
  -F "tts_text=你好，这是一个测试。" \
  -F "prompt_text=这是一个友好的声音" \
  -F "prompt_wav=@reference.wav" \
  -F "stream=true"
```

#### 响应格式

**非流式响应**:
```http
HTTP/1.1 200 OK
Content-Type: application/octet-stream
Transfer-Encoding: chunked

[二进制音频数据]
```

**流式响应**:
```http
HTTP/1.1 200 OK
Content-Type: application/octet-stream
Transfer-Encoding: chunked

[分块音频数据流]
```

#### 错误响应

```json
{
  "message": "Error description",
  "detail": "Detailed error information"
}
```

#### 代码示例

**Python 同步客户端**:
```python
import requests

def synthesize_speech(text, prompt_text, audio_file, stream=False):
    url = "http://localhost:8080/inference_zero_shot"

    with open(audio_file, 'rb') as f:
        files = {'prompt_wav': f}
        data = {
            'tts_text': text,
            'prompt_text': prompt_text,
            'stream': stream
        }

        response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            # 保存音频
            with open('output.wav', 'wb') as out_f:
                out_f.write(response.content)
            return True
        else:
            print(f"Error: {response.json()}")
            return False

# 使用示例
synthesize_speech(
    "你好世界",
    "友好自然的声音",
    "reference.wav"
)
```

**Python 异步客户端**:
```python
import aiohttp
import asyncio

async def async_synthesize(text, prompt_text, audio_file):
    url = "http://localhost:8080/inference_zero_shot"

    async with aiohttp.ClientSession() as session:
        with open(audio_file, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('tts_text', text)
            data.add_field('prompt_text', prompt_text)
            data.add_field('prompt_wav', f,
                          filename='reference.wav',
                          content_type='audio/wav')
            data.add_field('stream', 'false')

            async with session.post(url, data=data) as response:
                if response.status == 200:
                    audio_data = await response.read()
                    with open('output.wav', 'wb') as out_f:
                        out_f.write(audio_data)
                    return True
                else:
                    error = await response.json()
                    print(f"Error: {error}")
                    return False

# 运行示例
asyncio.run(async_synthesize("测试文本", "描述", "reference.wav"))
```

### 2. 查询TTS模型

#### 端点
```
GET /query_tts_model
POST /query_tts_model
```

#### 响应格式

```json
{
  "tts_models": [
    "CosyVoice2"
  ]
}
```

#### 请求示例

```bash
curl http://localhost:8080/query_tts_model
```

## WebSocket API

### 双向流式语音合成

#### 端点
```
WebSocket /inference_zero_shot_bistream
```

#### 通信协议

1. **连接建立**
2. **发送初始化参数**
3. **发送音频数据**
4. **发送文本流**
5. **接收音频流**

#### 消息格式

**初始化参数** (JSON):
```json
{
  "prompt_text": "这是一个友好的声音",
  "tts_model_name": "CosyVoice2"
}
```

**音频数据** (Binary):
- 参考音频的二进制数据 (WAV格式)

**文本请求** (JSON):
```json
{
  "tts_text": "要合成的文本",
  "finish": false
}
```

**结束请求** (JSON):
```json
{
  "tts_text": "最后一段文本",
  "finish": true
}
```

**音频响应** (Binary):
- 合成的音频数据流 (16位PCM)

#### Python 客户端示例

```python
import asyncio
import websockets
import json
import base64

async def bidirectional_tts():
    uri = "ws://localhost:8080/inference_zero_shot_bistream"

    async with websockets.connect(uri) as websocket:
        # 1. 发送初始化参数
        init_params = {
            "prompt_text": "友好自然的声音",
            "tts_model_name": "CosyVoice2"
        }
        await websocket.send(json.dumps(init_params))

        # 2. 发送参考音频
        with open("reference.wav", "rb") as f:
            audio_data = f.read()
        await websocket.send(audio_data)

        # 3. 发送文本并接收音频
        texts = ["你好", "这是一个测试", "再见"]

        for i, text in enumerate(texts):
            is_last = (i == len(texts) - 1)

            # 发送文本
            message = {
                "tts_text": text,
                "finish": is_last
            }
            await websocket.send(json.dumps(message))

            # 接收音频数据
            try:
                audio_chunk = await websocket.recv()
                # 处理音频数据...
                print(f"Received audio chunk for: {text}")
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed")
                break

# 运行客户端
asyncio.run(bidirectional_tts())
```

#### JavaScript 客户端示例

```javascript
class BidirectionalTTS {
  constructor(url) {
    this.url = url;
    this.websocket = null;
  }

  async connect(promptText, audioFile, modelName = "CosyVoice2") {
    this.websocket = new WebSocket(this.url);

    return new Promise((resolve, reject) => {
      this.websocket.onopen = async () => {
        try {
          // 发送初始化参数
          const initParams = {
            prompt_text: promptText,
            tts_model_name: modelName
          };
          this.websocket.send(JSON.stringify(initParams));

          // 发送参考音频
          const audioData = await this.readFileAsArrayBuffer(audioFile);
          this.websocket.send(audioData);

          resolve();
        } catch (error) {
          reject(error);
        }
      };

      this.websocket.onerror = (error) => reject(error);
    });
  }

  async synthesize(text, isLast = false) {
    const message = {
      tts_text: text,
      finish: isLast
    };
    this.websocket.send(JSON.stringify(message));
  }

  onAudio(callback) {
    this.websocket.onmessage = (event) => {
      if (event.data instanceof Blob) {
        callback(event.data);
      }
    };
  }

  readFileAsArrayBuffer(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsArrayBuffer(file);
    });
  }

  close() {
    if (this.websocket) {
      this.websocket.close();
    }
  }
}

// 使用示例
async function main() {
  const tts = new BidirectionalTTS("ws://localhost:8080/inference_zero_shot_bistream");

  try {
    await tts.connect("友好自然的声音", "reference.wav");

    tts.onAudio((audioBlob) => {
      // 处理接收到的音频
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      audio.play();
    });

    // 发送文本流
    await tts.synthesize("你好");
    await tts.synthesize("这是一个测试");
    await tts.synthesize("再见", true);

  } catch (error) {
    console.error("Error:", error);
  }
}

main();
```

## 系统端点

### 健康检查

#### 端点
```
GET /healthz
GET /health
GET /liveness
GET /readiness
```

#### 响应格式

```json
{
  "message": "Ok"
}
```

#### 错误响应
```json
{
  "message": "Error"
}
```

### Prometheus 指标

#### 端点
```
GET /metrics
```

#### 可用指标

| 指标名称 | 类型 | 描述 |
|----------|------|------|
| `lightllm_request_count` | Counter | 总请求数 |
| `lightllm_request_failure` | Counter | 失败请求数 |
| `lightllm_request_success` | Counter | 成功请求数 |
| `lightllm_request_latency` | Histogram | 请求延迟分布 |

## 错误处理

### HTTP 状态码

| 状态码 | 描述 | 示例场景 |
|---------|------|----------|
| 200 | 成功 | 音频合成完成 |
| 400 | 请求错误 | 参数缺失或格式错误 |
| 404 | 资源不存在 | 端点不存在 |
| 413 | 文件过大 | 音频文件超过限制 |
| 422 | 参数验证失败 | 文本长度超限 |
| 500 | 服务器错误 | 模型加载失败 |
| 503 | 服务不可用 | 系统资源不足 |

### 错误响应格式

```json
{
  "message": "Error description",
  "detail": "Detailed error information",
  "error_code": "INVALID_PARAMETER"
}
```

### 常见错误

1. **文件格式错误**
```json
{
  "message": "Unsupported audio format",
  "detail": "Only WAV format is supported"
}
```

2. **文本长度超限**
```json
{
  "message": "Text too long",
  "detail": "Maximum text length is 1000 characters"
}
```

3. **模型未加载**
```json
{
  "message": "Model not ready",
  "detail": "TTS model is still loading"
}
```

## 限流和配额

### 默认限制

| 限制项 | 默认值 | 说明 |
|--------|--------|------|
| 文件大小 | 50MB | 单个音频文件最大 |
| 文本长度 | 1000字符 | 单次合成最大 |
| 并发连接 | 100 | WebSocket最大并发 |
| 请求频率 | 10/分钟 | 单IP限流 |

### 自定义限制

可通过配置调整：

```python
# wsgi_wrapper.py
CONFIG = {
    "max_file_size": 100 * 1024 * 1024,  # 100MB
    "max_text_length": 2000,              # 2000字符
    "max_concurrent_requests": 200,       # 200并发
    "rate_limit": "100/hour",             # 每小时100次
}
```

## SDK 示例

### Python SDK

```python
from light_tts_client import LightTTSClient

# 初始化客户端
client = LightTTSClient(
    base_url="http://localhost:8080",
    timeout=30
)

# 同步合成
audio_data = client.synthesize(
    text="你好世界",
    prompt_text="友好声音",
    reference_audio="reference.wav"
)

# 保存音频
with open("output.wav", "wb") as f:
    f.write(audio_data)

# 流式合成
for chunk in client.synthesize_stream(
    text="长文本内容...",
    prompt_text="自然声音",
    reference_audio="reference.wav"
):
    # 处理音频流
    process_audio_chunk(chunk)
```

### Node.js SDK

```javascript
const { LightTTSClient } = require('@light-tts/client');

const client = new LightTTSClient({
  baseUrl: 'http://localhost:8080',
  timeout: 30000
});

async function synthesize() {
  try {
    const audioBuffer = await client.synthesize({
      text: 'Hello World',
      promptText: 'Friendly voice',
      referenceAudio: fs.readFileSync('reference.wav')
    });

    fs.writeFileSync('output.wav', audioBuffer);
  } catch (error) {
    console.error('Synthesis failed:', error);
  }
}
```