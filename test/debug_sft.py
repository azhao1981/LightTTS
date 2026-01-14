"""调试 SFT 音色检查"""
import sys
sys.path.insert(0, '/home/weiz/projects/ai/tts/LightTTS')

import torch
import yaml
from pathlib import Path

# 加载 spk2info.pt
model_dir = './pretrained_models/CosyVoice2-0.5B-finetune-v1'
spk2info_path = f'{model_dir}/spk2info.pt'
spk2info = torch.load(spk2info_path, map_location='cpu')
print("=== spk2info.pt 中的音色 ===")
for k in spk2info.keys():
    print(f"  - {k}: {list(spk2info[k].keys())}")

# 加载 voices.yaml
yaml_path = './voices.yaml'
if Path(yaml_path).exists():
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print("\n=== voices.yaml 中的音色 ===")
    if config and 'voices' in config:
        for v in config['voices']:
            print(f"  - {v.get('name', '')}")
else:
    print(f"\n❌ voices.yaml 不存在于: {yaml_path}")

# 模拟 is_valid_spk_id 逻辑
print("\n=== 模拟 is_valid_spk_id 检查 ===")

def check_is_valid_spk_id(base_spk_id, voices_dict, spk2info_dict):
    """模拟 speaker_manager.is_valid_spk_id"""
    print(f"检查 base_spk_id='{base_spk_id}'")
    print(f"  - in voices: {base_spk_id in voices_dict}")
    print(f"  - in spk2info: {base_spk_id in spk2info_dict}")
    return base_spk_id in voices_dict or base_spk_id in spk2info_dict

# 构造 voices 字典
voices_dict = {}
if Path(yaml_path).exists():
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if config and 'voices' in config:
        for v in config['voices']:
            voices_dict[v.get('name', '')] = v

print(f"检查 'female_test': {check_is_valid_spk_id('female_test', voices_dict, spk2info)}")
print(f"检查 'male1_trained': {check_is_valid_spk_id('male1_trained', voices_dict, spk2info)}")
print(f"检查 'female': {check_is_valid_spk_id('female', voices_dict, spk2info)}")
