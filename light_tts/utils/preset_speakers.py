import hashlib
from pathlib import Path
import torch
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

def warmup_presets(httpserver_manager, frontend=None):
    warmed_presets = {}

    logger.info("=" * 60)
    logger.info("Starting preset speakers warmup...")
    logger.info("=" * 60)

    for speaker_id, config in PRESET_SPEAKERS.items():
        audio_path = config['audio_path']
        logger.info(f"[{speaker_id}] Loading preset speaker from: {audio_path}")

        if not Path(audio_path).exists():
            logger.error(f"[{speaker_id}] Audio file not found: {audio_path}")
            logger.warning(f"Preset speaker '{speaker_id}' audio not found: {audio_path}")
            continue

        try:
            logger.info(f"[{speaker_id}] Loading WAV file...")
            prompt_speech_16k = load_wav(audio_path, 16000)
            logger.info(f"[{speaker_id}] WAV shape: {prompt_speech_16k.shape}")

            with open(audio_path, 'rb') as f:
                speech_md5 = hashlib.md5(f.read()).hexdigest()
            logger.info(f"[{speaker_id}] MD5: {speech_md5}")

            logger.info(f"[{speaker_id}] Allocating shared memory...")
            speech_index, have_alloc = httpserver_manager.alloc_speech_mem(speech_md5, prompt_speech_16k)
            logger.info(f"[{speaker_id}] Speech index: {speech_index}, have_alloc: {have_alloc}")

            # 检查共享内存状态
            use_mark = httpserver_manager.shared_speech_manager.use_marks.arr[speech_index]
            logger.info(f"[{speaker_id}] Shared memory use_mark after alloc: {use_mark} (0=free, 1=allocated, 2=data_set, 3=ready)")

            # 如果是新分配的，需要提取语音特征
            if not have_alloc:
                logger.info(f"[{speaker_id}] Extracting speech features...")

                # 如果提供了 frontend，使用它提取特征
                if frontend is not None:
                    logger.info(f"[{speaker_id}] Using provided frontend for feature extraction")

                    # 将 numpy 数组转换为 torch tensor
                    prompt_speech_16k_tensor = torch.from_numpy(prompt_speech_16k)

                    # 调用 frontend 提取特征
                    model_input = frontend.frontend_zero_shot(
                        '', '', prompt_speech_16k_tensor, 16000, ''
                    )

                    speech_token = model_input["llm_prompt_speech_token"].cpu().numpy()
                    speech_feat = model_input["prompt_speech_feat"].squeeze(0).cpu().numpy()
                    embedding = model_input["llm_embedding"].cpu().numpy()

                    # 设置完整的语音特征（这会将 use_mark 设置为 3）
                    httpserver_manager.shared_speech_manager.set_index_speech(
                        speech_index, speech_token, speech_feat, embedding
                    )

                    # 验证状态
                    use_mark_after = httpserver_manager.shared_speech_manager.use_marks.arr[speech_index]
                    logger.info(f"[{speaker_id}] ✅ Speech features extracted successfully, use_mark: {use_mark_after} (expected: 3)")

                    if use_mark_after != 3:
                        logger.error(f"[{speaker_id}] ❌ Unexpected use_mark after feature extraction: {use_mark_after}, expected 3")
                else:
                    logger.warning(f"[{speaker_id}] ⚠️ No frontend provided, speech features will be extracted on first request")
                    logger.warning(f"[{speaker_id}] This may cause issues if multiple requests use the same preset speaker simultaneously")
            else:
                logger.info(f"[{speaker_id}] Using cached speech data (have_alloc=True)")

            warmed_presets[speaker_id] = {
                "speech_md5": speech_md5,
                "speech_index": speech_index,
                "semantic_len": (prompt_speech_16k.shape[1] + 239) // 640 + 10,
                "prompt_text": config['prompt_text']
            }
            logger.info(f"[{speaker_id}] ✅ Preset speaker loaded successfully")
        except Exception as e:
            logger.error(f"[{speaker_id}] ❌ Failed to load preset speaker: {e}")
            import traceback
            logger.error(traceback.format_exc())

    logger.info("=" * 60)
    logger.info(f"Preset speakers warmup completed. Loaded {len(warmed_presets)}/{len(PRESET_SPEAKERS)} speakers")
    for speaker_id, info in warmed_presets.items():
        logger.info(f"  - {speaker_id}: index={info['speech_index']}, md5={info['speech_md5'][:8]}...")
    logger.info("=" * 60)

    return warmed_presets
