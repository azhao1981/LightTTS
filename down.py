#验证 ModelScope token
from modelscope import snapshot_download
from modelscope.hub.api import HubApi
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
api = HubApi()
token = os.getenv('MODELSCOPE_TOKEN')

api.login(token)
print(token)
#模型下载
snapshot_download('azhao2050/CosyVoice2-0.5B-finetune-v1', local_dir='pretrained_models/CosyVoice2-0.5B-finetune-v1')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')