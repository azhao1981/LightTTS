import torch
spk2info = torch.load('pretrained_models/CosyVoice2-0.5B-finetune-v1/spk2info.pt', map_location='cpu')

print('=== 可用音色列表 ===')
for i, (spk_id, info) in enumerate(spk2info.items()):
    print(f'{i+1}. {spk_id}')
    print(f'   字段: {list(info.keys())}')
    if 'llm_embedding' in info:
        print(f'   embedding 形状: {info["llm_embedding"].shape}')
    print(f'   prompt_speech_token: {"llm_prompt_speech_token" in info}')
    print(f'   prompt_speech_feat: {"prompt_speech_feat" in info}')
    print()