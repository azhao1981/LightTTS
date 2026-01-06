# import faulthandler
# faulthandler.enable()
import os
import numpy as np
import multiprocessing as mp
import threading
from multiprocessing import shared_memory
from light_tts.utils.log_utils import init_logger
from filelock import FileLock
from collections import OrderedDict

logger = init_logger(__name__)


class SharedArray:
    def __init__(self, name, shape, dtype):
        dtype_byte_num = np.array([1], dtype=dtype).dtype.itemsize
        dest_size = np.prod(shape) * dtype_byte_num
        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=dest_size)
            logger.info(f"create shm {name}")
        except Exception as e:
            shm = shared_memory.SharedMemory(name=name, create=False, size=dest_size)
            logger.info(f"link shm {name} error {str(e)}")
        
        if shm.size != dest_size:
            logger.info(f"size not same, unlink shm {name} and create again")
            shm.unlink()
            shm.close()
            try:
                shm = shared_memory.SharedMemory(name=name, create=True, size=dest_size)
                logger.info(f"create shm {name}")
            except Exception as e:
                shm = shared_memory.SharedMemory(name=name, create=False, size=dest_size)
                logger.info(f"link shm {name} error {str(e)}")

        self.shm = shm  # SharedMemory 对象一定要被持有，否则会被释放
        self.arr = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)


class SharedTensorManager:
    def __init__(self, name, size) -> None:
        self.name = name
        self.size = size
        self.shape_infs = SharedArray(f"{name}_shapes", (size, 2), dtype=np.int32)
        self.tensors = [None for _ in range(size)]
        return
    
    def set_index_data(self, index, shape, data, dtype):
        shm_arr = SharedArray(f"{self.name}_{index}_tensor", shape, dtype=dtype)
        shm_arr.arr[:, :] = data
        self.shape_infs.arr[index,:] = shape
        self.tensors[index] = shm_arr
        return
    
    def get_index_tensor_shape(self, index):
        return tuple(self.shape_infs.arr[index])
    
    def get_index_tensor(self, index, dtype):
        shape = self.get_index_tensor_shape(index)
        shm_arr = SharedArray(f"{self.name}_{index}_tensor", shape, dtype=dtype)
        self.tensors[index] = shm_arr
        return shm_arr
    
    # def release(self, index):
    #     shape = self.get_index_tensor_shape(index)
    #     shm_arr = SharedArray(f"{self.name}_{index}_tensor", shape, dtype=np.float16)
    #     shm_arr.shm.unlink() # 销毁shm。
    #     shm_arr.shm.close()
    #     logger.info(f"release shm tensor index {index}")
    #     return


class SharedSpeechManager:
    def __init__(self, name, size, init_mark=True) -> None:
        self.name = name
        self.size = size
        self.use_marks = SharedArray(f"{name}_use_marks", (size,), dtype=np.int32)
        if init_mark:
            self.use_marks.arr[:] = 0
        self.lru_cache = OrderedDict()
        self.lock = threading.Lock()

        # 新增: spk_id 到 speech_index 的映射 (用于预设音色快速路径)
        self.spk_id_to_index = {}

        self.prompt_speech_16k_manager = SharedTensorManager(f"{name}_prompt_speech_16k", size)
        self.speech_feat_manager = SharedTensorManager(f"{name}_speech_feat", size)
        self.speech_token_manager = SharedTensorManager(f"{name}_speech_token", size)
        self.spk_embedding_manager = SharedTensorManager(f"{name}_spk_embedding", size)
        return
        
    
    def alloc(self, speech_md5):
        with self.lock:
            if speech_md5 in self.lru_cache:
                self.lru_cache.move_to_end(speech_md5)
                return self.lru_cache[speech_md5], True
            index = None
            if len(self.lru_cache) >= self.size:
                key, value = self.lru_cache.popitem(last=False)
                index = value
            else:
                for i in range(self.size):
                    if self.use_marks.arr[i] == 0:
                        index = i
                        break

            if index is None:
                raise RuntimeError("alloc error")

            self.use_marks.arr[index] = 1
            self.lru_cache[speech_md5] = index
            return index, False

    def alloc_by_spk_id(self, spk_id):
        """
        通过 spk_id 分配共享内存 (预设音色快速路径)

        Args:
            spk_id: 音色 ID

        Returns:
            (index, have_alloc): index=共享内存索引, have_alloc=是否已缓存
        """
        with self.lock:
            # 检查 spk_id 是否已映射
            if spk_id in self.spk_id_to_index:
                index = self.spk_id_to_index[spk_id]
                # 更新 LRU (使用 spk_id 作为 key)
                if spk_id in self.lru_cache:
                    self.lru_cache.move_to_end(spk_id)
                else:
                    # 如果 spk_id 不在 lru_cache 中,添加它
                    # 注意: 这可能发生在预加载时 spk_id 已映射但未在 lru_cache 中的情况
                    self.lru_cache[spk_id] = index
                return index, True  # 命中缓存

            # 未映射,分配新索引
            index = None
            if len(self.lru_cache) >= self.size:
                # 缓存已满,驱逐最久未使用的项
                key, value = self.lru_cache.popitem(last=False)
                # 如果驱逐的是 spk_id,从 spk_id_to_index 中移除
                if key in self.spk_id_to_index:
                    del self.spk_id_to_index[key]
                index = value
            else:
                # 找到空闲槽位
                for i in range(self.size):
                    if self.use_marks.arr[i] == 0:
                        index = i
                        break

            if index is None:
                raise RuntimeError(f"alloc_by_spk_id failed: no available slot for spk_id={spk_id}")

            # 标记为已分配
            self.use_marks.arr[index] = 1
            # 建立双重映射: spk_id → index 和 index → spk_id (通过 lru_cache)
            self.spk_id_to_index[spk_id] = index
            self.lru_cache[spk_id] = index

            return index, False  # 新分配

    def set_index_data(self, index, shape, data):
        self.prompt_speech_16k_manager.set_index_data(index, shape, data, np.float32)
        self.use_marks.arr[index] = 2
        return

    def get_index_data(self, index):
        if self.use_marks.arr[index] >= 2:
            return self.prompt_speech_16k_manager.get_index_tensor(index, dtype=np.float32)
        return None

    def set_index_speech(self, index, speech_token, speech_feat, spk_embedding):
        self.speech_token_manager.set_index_data(index, speech_token.shape, speech_token, np.int32)
        self.speech_feat_manager.set_index_data(index, speech_feat.shape, speech_feat, np.float32)
        self.spk_embedding_manager.set_index_data(index, spk_embedding.shape, spk_embedding, np.float32)
        self.use_marks.arr[index] = 3
        return

    def get_index_speech_token(self, index):
        if self.use_marks.arr[index] >= 3:
            return self.speech_token_manager.get_index_tensor(index, dtype=np.int32)
        return None

    def get_index_speech(self, index):
        if self.use_marks.arr[index] >= 3:
            return self.speech_token_manager.get_index_tensor(index, dtype=np.int32), self.speech_feat_manager.get_index_tensor(index, dtype=np.float32), self.spk_embedding_manager.get_index_tensor(index, np.float32)
        return None

    def speech_data_ready(self, index):
        if self.use_marks.arr[index] >= 3:
            return True
        return False