from src.config_reader import config
from .models import BaseModel, BertModel


def get_model() -> BaseModel:
    model: BaseModel = BaseModel()
    if config['model']['type'] == 'bert':
        model: BertModel = BertModel()

    return model

