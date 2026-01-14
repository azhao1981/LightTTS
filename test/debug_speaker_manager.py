"""独立测试 speaker_manager 的行为"""
import sys
import os
sys.path.insert(0, '/home/weiz/projects/ai/tts/LightTTS')

import torch
import yaml
from pathlib import Path
from light_tts.server.speaker_manager import SpeakerManager
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.utils.io import load_yaml

# 模拟服务器初始化
model_dir = './pretrained_models/CosyVoice2-0.5B-finetune-v1'
configs = load_yaml(model_dir)

print("=== 1. 初始化 CosyVoiceFrontEnd ===")
frontend = CosyVoiceFrontEnd(
    configs['get_tokenizer'],
    configs['feat_extractor'],
    f'{model_dir}/campplus.onnx',
    f'{model_dir}/speech_tokenizer_v2.onnx',
    f'{model_dir}/spk2info.pt',
    configs['allowed_special']
)
print(f"frontend.spk2info keys: {list(frontend.spk2info.keys())}")

print("\n=== 2. 初始化 SpeakerManager ===")
voices_yaml_path = './voices.yaml'
speaker_manager = SpeakerManager(model=frontend, yaml_path=voices_yaml_path)
speaker_manager.load_presets()
print(f"speaker_manager.voices keys: {list(speaker_manager.voices.keys())}")

print("\n=== 3. 测试 is_valid_spk_id ===")
test_cases = [
    'female',           # 在 voices.yaml
    'female_test',      # 在 spk2info.pt
    'male1_trained',    # 在 spk2info.pt
    'female_sft',       # SFT mode
    'female_test_sft',  # SFT mode
    'invalid_sft',      # 无效
]

for spk_id in test_cases:
    result = speaker_manager.is_valid_spk_id(spk_id)
    print(f"  is_valid_spk_id('{spk_id}'): {result}")

print("\n=== 4. 详细调试 SFT 检查 ===")
print("检查 'female_test_sft':")
base = 'female_test'
print(f"  - base_spk_id: {base}")
print(f"  - in voices: {base in speaker_manager.voices}")
print(f"  - in spk2info: {base in frontend.spk2info}")
print(f"  - hasattr(frontend, 'spk2info'): {hasattr(frontend, 'spk2info')}")
if hasattr(frontend, 'spk2info'):
    print(f"  - frontend.spk2info type: {type(frontend.spk2info)}")
    print(f"  - frontend.spk2info keys: {list(frontend.spk2info.keys())}")
