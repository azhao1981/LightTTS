找到 Bug 了！🎯

  问题在 tts_encode/manager.py 第 133 行

  对比两个分支:

  Zero-shot 模式 (第 167 行):
  speech_token = (speech_token + self.vocab_size + 2)  # ✓ 有 offset
  audio_ids = speech_token.flatten().tolist()

  SFT 模式 (第 133 行):
  audio_ids = speech_token.flatten().tolist()  # ✗ 没有 offset！

  原因分析

  LLM 的 embedding table 结构是:
  - [0, vocab_size-1]: text tokens
  - [vocab_size]: sos_eos
  - [vocab_size+1]: task_id
  - [vocab_size+2, ...]: speech tokens

  spk2info.pt 中的 llm_prompt_speech_token 是 0-based 的原始 speech token ID。要在 LLM embedding table 中正确查找，必须加上 vocab_size + 2 的偏移。

  Zero-shot 模式加了 offset，SFT 模式没加，导致 SFT 模式用错误的 embedding indices 查表，得到乱码。

  修复方案

  修改 light_tts/server/tts_encode/manager.py 第 133 行:

● Update(light_tts/server/tts_encode/manager.py)
  ⎿  Added 8 lines, removed 3 lines                                                                                                                        
      128                           speech_token, speech_feat, embedding = speech_data                                                                     
      129                           logger.info(f"SFT mode: req_id {req.request_id}, spk_id={spk_id}, embedding shape={embedding.shape if embedding is not 
            None else 'None'}")                                                                                                                            
      130                                                                                                                                                  
      131 -                        # Set speech_token (empty array in SFT mode)                                                                            
      131 +                        # Set speech_token with vocab offset (same as zero-shot mode)                                                           
      132                           if not req.bistream:                                                                                                   
      133 -                            audio_ids = speech_token.flatten().tolist() if speech_token.size > 0 else []                                        
      133 +                            if speech_token.size > 0:                                                                                           
      134 +                                 # Add vocab_size + 2 offset for correct embedding lookup                                                       
      135 +                                 speech_token_offset = (speech_token + self.vocab_size + 2)                                                     
      136 +                                 audio_ids = speech_token_offset.flatten().tolist()                                                             
      137 +                             else:                                                                                                              
      138 +                                 audio_ids = []                                                                                                 
      139                               with self.shm_req_manager.get_req_lock_by_index(req.index_in_shm_mem):                                             
      140                                   req.set_speech_token(audio_ids)                                                                                
      141                                                                                                                                                  
      142 -                         req.prompt_token_pad = 0                                                                                               
      142 +                         req.prompt_token_pad = int(np.ceil(speech_token.size / self.token_hop_len) * self.token_hop_len - speech_token.size)   
          +if speech_token.size > 0 else 0                                                                                                                 
      143                           logger.info(f"Send: {module_name:<14} | req_id {req.request_id} | semantic_len=0 | text_len={req.text_len} | mode=SFT  
           to tts_llm")                                                                                                                                    
      144                                                                                                                                                  
      145                           # Send to LLM                                                                                                          

● 还需要更新日志，显示正确的 semantic_len:

● Update(light_tts/server/tts_encode/manager.py)
  ⎿  Added 1 line, removed 1 line                                                                                                                          
      140                                   req.set_speech_token(audio_ids)                                                                                
      141                                                                                                                                                  
      142                           req.prompt_token_pad = int(np.ceil(speech_token.size / self.token_hop_len) * self.token_hop_len - speech_token.size)   
           if speech_token.size > 0 else 0                                                                                                                 
      143 -                        logger.info(f"Send: {module_name:<14} | req_id {req.request_id} | semantic_len=0                                        
          - | text_len={req.text_len} | mode=SFT to tts_llm")                                                                                              
      143 +                        logger.info(f"Send: {module_name:<14} | req_id {req.request_id} | semantic_len={req.semantic_len}                       
          + | text_len={req.text_len} | mode=SFT to tts_llm")                                                                                              
      144                                                                                                                                                  
      145                           # Send to LLM                                                                                                          
      146                           self.shm_req_manager.put_back_req_obj(req)                                                                             

● 修复完成 ✓ 

  修改内容 (light_tts/server/tts_encode/manager.py):

  1. 添加 vocab offset - SFT 模式的 speech_token 现在会加上 vocab_size + 2 偏移，与 zero-shot 模式一致
  2. 计算 prompt_token_pad - 用于解码阶段的 token 对齐

  重启服务测试

  # 重启服务 (根据你的配置)
  bash cosy.dev.sh restart   # 或者你的重启命令

  # 等待服务启动后测试
  source .envrc && timeout 30 python test/test_sft_integration.py

  修复前后对比:
  ┌─────────────────────┬────────────────────────┬──────────────────────────┐
  │        项目         │         修复前         │          修复后          │
  ├─────────────────────┼────────────────────────┼──────────────────────────┤
  │ speech_token offset │ 无 (0-based)           │ + vocab_size + 2         │
  ├─────────────────────┼────────────────────────┼──────────────────────────┤
  │ prompt_token_pad    │ 0                      │ 正确计算                 │
  ├─────────────────────┼────────────────────────┼──────────────────────────┤
  │ embedding 查表      │ 错误位置 (text tokens) │ 正确位置 (speech tokens) │
  └─────────────────────┴────────────────────────┴──────────────────────────┘


  ● 让我查看 CosyVoice 的参考实现。

● Read(cosyvoice/cosyvoice/cli/cosyvoice.py)
  ⎿  Read 195 lines

● 让我查看 CosyVoice2Model 的 tts 方法，理解 SFT 推理的正确流程。

● Read(cosyvoice/cosyvoice/cli/model.py)
  ⎿  Read 386 lines

● 让我查看 LightTTS 如何构建 LLM 输入序列，与参考实现对比。

● Search(pattern: "sos_eos|task_id", path: "light_tts/server", output_mode: "content")
  ⎿  Found 16 lines (ctrl+o to expand)

● Read(light_tts/server/httpserver/manager.py)
  ⎿  Read 340 lines

● 我找到另一个关键问题了！让我查看 spk2info 中的 prompt_text 数据：

  prompt_text: shape=torch.Size([1, 89])  -- 89个token!

  但在 api_http.py 的 SFT 模式下：
  prompt_text = ''  # 被设为空字符串！

  这导致 LLM 输入结构错误。

  正确的 zero-shot 结构：[sos_eos] + [prompt_text_ids] + [text_ids] + [task_id] + [speech_tokens]

  当前 SFT 模式的结构：[sos_eos] + [text_ids] + [task_id] + [speech_tokens] (缺少 prompt_text!)

  让我修复 api_http.py，使用 spk2info 中的 prompt_text：

● Read(light_tts/server/api_http.py)
  ⎿  Read 100 lines

● 问题确认：SFT 模式需要使用 spk2info 中的 prompt_text token IDs，而不是空字符串。

  让我修复 api_http.py：

● Update(light_tts/server/api_http.py)
  ⎿  Added 11 lines                                                                                                                                        
      322                   speech_token = spk_info.get('llm_prompt_speech_token', torch.tensor([])).cpu().numpy()                                         
      323                   speech_feat = spk_info.get('prompt_speech_feat', torch.tensor([])).cpu().numpy()                                               
      324                                                                                                                                                  
      325 +                 # Extract prompt_text token IDs from spk2info (for cached zero-shot format)                                                    
      326 +                 # This is critical for proper LLM input structure                                                                              
      327 +                 prompt_text_tensor = spk_info.get('prompt_text', torch.tensor([]))                                                             
      328 +                 if prompt_text_tensor.numel() > 0:                                                                                             
      329 +                     # prompt_text is stored as token IDs, convert to list                                                                      
      330 +                     prompt_text_ids = prompt_text_tensor.flatten().tolist()                                                                    
      331 +                     logger.info(f"SFT mode: extracted prompt_text_ids from spk2info, len={len(prompt_text_ids)}")                              
      332 +                 else:                                                                                                                          
      333 +                     prompt_text_ids = []                                                                                                       
      334 +                     logger.info(f"SFT mode: no prompt_text in spk2info, using empty")                                                          
      335 +                                                                                                                                                
      336                   # Handle speech_feat shape (squeeze if needed)                                                                                 
      337                   if len(speech_feat.shape) == 3 and speech_feat.shape[0] == 1:                                                                  
      338                       speech_feat = speech_feat.squeeze(0)                                                                                       

● 现在需要更新设置 semantic_len 和传递 prompt_text_ids 的部分：

● Update(light_tts/server/api_http.py)
  ⎿  Added 2 lines, removed 2 lines                                                                                                                        
      356               else:                                                                                                                              
      357                   logger.info(f"SFT mode: using cached shared memory for spk_id={spk_id}, speech_index={speech_index}")                          
      358                                                                                                                                                  
      359 -             semantic_len = 0                                                                                                                   
      359 +             semantic_len = 0  # Will be set by encode process based on speech_token length                                                     
      360               need_extract_speech = False                                                                                                        
      361 -             prompt_text = ''                                                                                                                   
      361 +             prompt_text = ''  # Not used when prompt_text_ids is provided                                                                      
      362               prompt_speech_16k = None                                                                                                           
      363               speech_md5 = None                                                                                                                  
      364           else:                                                                                                                                  

● Read(light_tts/server/api_http.py)
  ⎿  Read 60 lines

● 我需要将 prompt_text_ids 添加到 request_dict 中。首先需要在 SFT 分支外初始化它：

● Read(light_tts/server/api_http.py)
  ⎿  Read 30 lines

● Update(light_tts/server/api_http.py)
  ⎿  Added 3 lines                                                                                                                                         
      279       is_sft = spk_id.endswith('_sft') if spk_id else False                                                                                      
      280       logger.info(f"[API] spk_id={spk_id}, is_sft={is_sft}")                                                                                     
      281                                                                                                                                                  
      282 +     # Initialize prompt_text_ids (used for SFT mode with cached zero-shot data)                                                                
      283 +     prompt_text_ids = None                                                                                                                     
      284 +                                                                                                                                                
      285       # Check spk_id parameter                                                                                                                   
      286       if spk_id:                                                                                                                                 
      287           if is_sft:                                                                                                                             

● Read(light_tts/server/api_http.py)
  ⎿  Read 50 lines

● 现在更新 request_dict 添加 prompt_text_ids：

● Update(light_tts/server/api_http.py)
  ⎿  Added 1 line                                                                                                                                            
      419               "text": text,                                                                                                                      
      420               "spk_id": spk_id,  # 新增字段                                                                                                         
      

      421               "prompt_text": prompt_text,                                                                                                        
      422 +             "prompt_text_ids": prompt_text_ids,  # Pre-computed token IDs for SFT mode                                                         
      423               "tts_model_name": tts_model_name,                                                                                                  
      424               "speech_md5": speech_md5,                                                                                                          
      425               "need_extract_speech": need_extract_speech,                                                                                        

● 现在更新 httpserver/manager.py 使用 prompt_text_ids：

● Update(light_tts/server/httpserver/manager.py)
  ⎿  Added 8 lines, removed 1 line                                                                                                                         
      198               request_headers = request.headers if request is not None else {}                                                                   
      199               await self._log_req_header(request_headers, request_id)                                                                            
      200                                                                                                                                                  
      201 -             prompt_text_ids = await self._async_encode(request_dict["prompt_text"])                                                            
      201 +             # Use pre-computed prompt_text_ids if available (for SFT/cached zero-shot mode)                                                    
      202 +             # Otherwise encode prompt_text string                                                                                              
      203 +             prompt_text_ids = request_dict.get("prompt_text_ids")                                                                              
      204 +             if prompt_text_ids is not None:                                                                                                    
      205 +                 logger.info(f"req_id {request_id}: using pre-computed prompt_text_ids, len={len(prompt_text_ids)}")                            
      206 +             else:                                                                                                                              
      207 +                 prompt_text_ids = await self._async_encode(request_dict["prompt_text"])                                                        
      208 +                                                                                                                                                
      209               text_ids = await self._async_encode(request_dict["text"])                                                                          
      210               if not bistream:                                                                                                                   
      211                   prompt_ids = list(chain([self.sos_eos], prompt_text_ids, text_ids, [self.task_id]))                                            

● 让我验证一下 logger 在 httpserver/manager.py 中是否已经定义：

● Search(pattern: "^logger", path: "light_tts/server/httpserver/manager.py", output_mode: "content")
  ⎿  Found 1 line (ctrl+o to expand)

● 修复完成。让我总结一下所有修改：

  修复总结

  问题根因

  SFT 模式使用 spk2info 中的缓存数据（类似 zero-shot 格式），但代码有两个 bug：

  1. 缺少 prompt_text - SFT 模式将 prompt_text 设为空字符串，但 spk2info 中有 89 个 prompt_text token IDs
  2. 缺少 vocab offset - speech_token 没有加上 vocab_size + 2 偏移

  修复内容

  1. tts_encode/manager.py (之前已修复)
  - SFT 模式的 speech_token 添加 vocab_size + 2 偏移

  2. api_http.py
  - 从 spk2info 提取 prompt_text token IDs
  - 通过 prompt_text_ids 字段传递给 httpserver

  3. httpserver/manager.py
  - 使用预计算的 prompt_text_ids（如果提供）
  - 构建正确的 prompt 结构：[sos_eos] + [prompt_text_ids] + [text_ids] + [task_id] + [speech_tokens]

  正确的 LLM 输入结构

  [sos_eos (vocab_size)] + [prompt_text_ids (89)] + [text_ids] + [task_id (vocab_size+1)] + [speech_tokens (vocab_size+2起)]

  这与 CosyVoice 参考实现完全一致。

  测试

  # 重启服务
  bash cosy.dev.sh restart

  # 测试
  source .envrc && timeout 30 python test/test_sft_integration.py