# Copyright (c) 2024 LightTTS Team
# Licensed under the Apache License, Version 2.0

"""
SpeakerManager: 管理 CosyVoice2 预设音色的加载和查询

功能:
1. 从 voices.yaml 读取音色配置
2. 启动时批量注册音色到 CosyVoice2 模型
3. 提供音色 ID 验证和查询接口
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import torch
from cosyvoice.utils.file_utils import load_wav


logger = logging.getLogger(__name__)


class SpeakerManager:
    """音色管理器"""

    def __init__(self, model, yaml_path: str = None, shared_speech_manager=None):
        """
        初始化音色管理器

        Args:
            model: CosyVoice2 模型实例
            yaml_path: voices.yaml 配置文件路径
            shared_speech_manager: 共享内存管理器 (用于预设音色预分配)
        """
        self.model = model
        self.yaml_path = yaml_path
        self.shared_speech_manager = shared_speech_manager
        self.voices: Dict[str, dict] = {}  # {spk_id: {audio_path, prompt_text, speech_index}}
        self.loaded_count = 0
        self.failed_count = 0

    def load_presets(self) -> bool:
        """
        从 YAML 加载并注册所有预设音色

        Returns:
            bool: 是否至少成功加载了一个音色
        """
        if not self.yaml_path or not os.path.exists(self.yaml_path):
            logger.warning(f"音色配置文件不存在: {self.yaml_path}, 跳过预设音色加载")
            return False

        try:
            with open(self.yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if not config or 'voices' not in config:
                logger.warning(f"配置文件格式错误,缺少 'voices' 字段: {self.yaml_path}")
                return False

            logger.info("=" * 60)
            logger.info("开始加载预设音色...")
            logger.info(f"配置文件: {self.yaml_path}")

            for voice_config in config['voices']:
                spk_id = voice_config.get('name', '')
                audio_path = voice_config.get('audio_path', '')
                prompt_text = voice_config.get('prompt_text', '')

                if not spk_id or not audio_path or not prompt_text:
                    logger.warning(f"跳过无效配置: {voice_config}")
                    self.failed_count += 1
                    continue

                success = self._register_voice(spk_id, audio_path, prompt_text)
                if success:
                    self.loaded_count += 1
                else:
                    self.failed_count += 1

            logger.info("-" * 60)
            logger.info(f"音色加载完成: 成功 {self.loaded_count} 个, 失败 {self.failed_count} 个")
            logger.info(f"可用音色 ID: {list(self.voices.keys())}")
            logger.info("=" * 60)

            return self.loaded_count > 0

        except Exception as e:
            logger.error(f"加载音色配置失败: {e}", exc_info=True)
            return False

    def _register_voice(self, spk_id: str, audio_path: str, prompt_text: str) -> bool:
        """
        注册单个音色到模型和共享内存

        Args:
            spk_id: 音色 ID
            audio_path: 音频文件路径 (相对于 YAML 或绝对路径)
            prompt_text: 提示文本

        Returns:
            bool: 是否注册成功
        """
        try:
            # 处理音频文件路径
            if not os.path.isabs(audio_path):
                yaml_dir = os.path.dirname(self.yaml_path)
                audio_path = os.path.join(yaml_dir, audio_path)

            if not os.path.exists(audio_path):
                logger.error(f"✗ 音频文件不存在: {audio_path} (spk_id: {spk_id})")
                return False

            # 加载音频文件
            prompt_speech_16k = load_wav(audio_path, 16000)

            # 直接调用 CosyVoiceFrontEnd 的 frontend_zero_shot 方法提取特征
            # 然后将结果存入 spk2info 字典 (等效于 model.add_zero_shot_spk)
            model_input = self.model.frontend_zero_shot('', prompt_text, prompt_speech_16k, 24000, '')
            del model_input['text']
            del model_input['text_len']

            # 存入 spk2info 字典
            self.model.spk2info[spk_id] = model_input

            # 如果有共享内存管理器,预分配共享内存并存储特征
            speech_index = None
            semantic_len = 0  # 默认值
            if self.shared_speech_manager is not None:
                # 使用 spk_id 直接分配共享内存 (无需 MD5)
                speech_index, have_alloc = self.shared_speech_manager.alloc_by_spk_id(spk_id)

                # 预计算语义长度 (与动态上传模式保持一致)
                semantic_len = (prompt_speech_16k.shape[1] + 239) // 640 + 10

                if not have_alloc:
                    # 第一次分配,存储原始音频数据到共享内存
                    self.shared_speech_manager.set_index_data(
                        speech_index,
                        prompt_speech_16k.shape,
                        prompt_speech_16k.cpu().numpy()
                    )

                    # 提取并存储特征到共享内存 (供 Encode 模块复用)
                    speech_token = model_input["llm_prompt_speech_token"].cpu().numpy()
                    speech_feat = model_input["prompt_speech_feat"].squeeze(0).cpu().numpy()
                    embedding = model_input["llm_embedding"].cpu().numpy()

                    self.shared_speech_manager.set_index_speech(
                        speech_index, speech_token, speech_feat, embedding
                    )

                    logger.info(f"✓ 加载并预分配共享内存: [{spk_id}] → speech_index={speech_index}, semantic_len={semantic_len}")
                else:
                    logger.info(f"✓ 加载 (已缓存): [{spk_id}] → speech_index={speech_index}, semantic_len={semantic_len}")

            self.voices[spk_id] = {
                'audio_path': audio_path,
                'prompt_text': prompt_text,
                'speech_index': speech_index,  # 存储共享内存索引
                'semantic_len': semantic_len,  # 存储预计算的语义长度
            }
            return True

        except Exception as e:
            logger.error(f"✗ 加载异常: [{spk_id}] - {e}", exc_info=True)
            return False

    def is_valid_spk_id(self, spk_id: str) -> bool:
        """检查音色 ID 是否有效"""
        return spk_id in self.voices

    def list_available_spks(self) -> List[str]:
        """获取所有可用的音色 ID"""
        return list(self.voices.keys())

    def get_voice_info(self, spk_id: str) -> Optional[dict]:
        """获取音色详细信息"""
        return self.voices.get(spk_id)

    def get_statistics(self) -> dict:
        """获取加载统计信息"""
        return {
            'loaded': self.loaded_count,
            'failed': self.failed_count,
            'total': self.loaded_count + self.failed_count,
            'available_spk_ids': list(self.voices.keys())
        }
