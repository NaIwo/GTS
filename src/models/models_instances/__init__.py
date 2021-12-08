import os
from src.utils import config

if config['general']['device'] == 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from .base_model import BaseModel
from .bert_model import BertModel
from .cnn_model import CnnModel
from .bilstm_model import BilstmModel
