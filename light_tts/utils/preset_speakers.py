import hashlib
from pathlib import Path
from cosyvoice.utils.file_utils import load_wav
from light_tts.utils.log_utils import init_logger

logger = init_logger(__name__)

PRESET_SPEAKERS = {
    "male": {
        "audio_path": "asset/xiangyu.wav",
        "prompt_text": "咱们这个项目呢，属于是投入低、回本快。而且现在加盟呢，还有一些政策上的这个优惠。呃，您看我后续让招商经理联系您，给您详细介绍一下可以吗？呃，不会占用您太多时间的，也是给您自己一个赚钱的机会嘛。",
    },
    "female": {
        "audio_path": "asset/wangye1.wav",
        "prompt_text": "使用费的话，这一块是属于你管品牌就三万块钱。培训的话，培训费要两万，然后设计费要三千，然后系统使用费要两两千块钱这样子。那这一次的话就是今年我们加盟它，这些都是减免的。",
    },
}

def warmup_presets(httpserver_manager):
    warmed_presets = {}

    for speaker_id, config in PRESET_SPEAKERS.items():
        audio_path = config['audio_path']

        if not Path(audio_path).exists():
            logger.warning(f"Preset speaker '{speaker_id}' audio not found: {audio_path}")
            continue

        try:
            prompt_speech_16k = load_wav(audio_path, 16000)
            with open(audio_path, 'rb') as f:
                speech_md5 = hashlib.md5(f.read()).hexdigest()
            speech_index, _ = httpserver_manager.alloc_speech_mem(speech_md5, prompt_speech_16k)

            warmed_presets[speaker_id] = {
                "speech_md5": speech_md5,
                "speech_index": speech_index,
                "semantic_len": (prompt_speech_16k.shape[1] + 239) // 640 + 10,
                "prompt_text": config['prompt_text']
            }
            logger.info(f"Preset speaker '{speaker_id}' loaded")
        except Exception as e:
            logger.error(f"Failed to load preset speaker '{speaker_id}': {e}")

    return warmed_presets
