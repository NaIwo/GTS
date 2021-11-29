from .models_instances import BaseModel, BertModel
from src.config_reader import config
from src.datasets import Dataset


from typing import Optional


class GtsModel:
    def __init__(self, model: BaseModel):
        self._model: BaseModel = model

    def __call__(self, inputs: Dataset, training: Optional[bool] = None, **kwargs):
        self._model(inputs, training, **kwargs)


if config['encoder']['type'] == 'bert':
    gts_model: GtsModel = GtsModel(BertModel())
elif config['encoder']['type'] == 'cnn':
    gts_model: GtsModel = GtsModel(BertModel())
elif config['encoder']['type'] == 'bilstm':
    gts_model: GtsModel = GtsModel(BertModel())
