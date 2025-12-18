
dir=$(cd "$(dirname "")"; pwd)
cd ~/tts && source .envrc                                                                                                                                      
export FINETURED_COSYVOICE2_MODEL_PATH="/root/.cache/modelscope/hub/models/azhao2050/CosyVoice2-0.5B-finetune-v1"
cd $dir
python -m light_tts.server.api_server \                                                                                                                        
  --model_dir "$FINETURED_COSYVOICE2_MODEL_PATH" \                                                                                                             
  --host 0.0.0.0 --port 8080 \                                                                                                                                 
  --load_trt True \                                                                                                                                            
  --flow_steps 5 \                                                                                                                                             
  --data_type bfloat16 \                                                                                                                                       
  --max_total_token_num 65536 \                                                                                                                                
  --max_req_total_len 32768 
