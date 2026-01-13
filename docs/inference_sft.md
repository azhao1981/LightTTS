需要修改的模块
  ┌─────────┬──────────────────────────┬──────────────────────────────────┐
  │  模块   │           文件           │              改动点              │
  ├─────────┼──────────────────────────┼──────────────────────────────────┤
  │ API     │ api_http.py              │ 添加 /inference_sft 端点         │
  ├─────────┼──────────────────────────┼──────────────────────────────────┤
  │ Manager │ httpserver/manager.py    │ 添加 SFT 请求处理分支            │
  ├─────────┼──────────────────────────┼──────────────────────────────────┤
  │ Encode  │ tts_encode/manager.py    │ 添加 SFT 特征提取逻辑            │
  ├─────────┼──────────────────────────┼──────────────────────────────────┤
  │ Core    │ core/objs/io_objs/req.py │ 添加 mode 字段区分 sft/zero_shot │
  └─────────┴──────────────────────────┴──────────────────────────────────┘
  核心差异

  SFT 模式 (frontend_sft, frontend.py:151-155):
  model_input = {
      'text': tts_text_token,
      'text_len': tts_text_token_len,
      'llm_embedding': spk2info[spk_id]['embedding'],
      'flow_embedding': spk2info[spk_id]['embedding']
  }
  # 只需要 4 个字段，无 prompt 信息

  Zero-shot 模式 (frontend_zero_shot, frontend.py:157-179):
  model_input = {
      'text': ...,
      'text_len': ...,
      'prompt_text': ...,
      'prompt_text_len': ...,
      'llm_prompt_speech_token': ...,  # 需要提取
      'prompt_speech_feat': ...,         # 需要提取
      'llm_embedding': ...,              # 需要提取
      'flow_embedding': ...
  }
  # 需要 8 个字段，需要完整特征提取

  实现要点

  1. API 层: 新增端点，参数改为 spk_id (字符串)，移除 prompt_wav, prompt_text
  2. HttpServerManager:
    - SFT 模式 prompt_ids = [sos_eos] + text_ids + [task_id]
    - 不需要 semantic_len (SFT 无语音 prompt)
  3. Encode Manager:
    - 添加分支：if req.mode == 'sft': embedding = frontend.spk2info[spk_id]['embedding']
    - 跳过 speech_token, speech_feat 提取
  4. 共享内存: SFT 模式不需要预分配音频内存

  工作量估算

  - 代码量: ~200 行
  - 测试: 需要验证 spk2info 中有对应的 SFT 音色
  - 风险: 低，逻辑独立，不影响现有 zero_shot 流程

  前置条件

  需要确认 spk2info.pt 文件中是否包含预训练的 SFT 音色 (如 中文女)。如果只有 zero-shot 音色，需要重新训练模型。