import os
from src.config_reader import config

if config['device'] == 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from .base_model import BaseModel
from .bert_model import BertModel
