from enum import Enum
from src.config_reader import config


class TagVectorID(Enum):
    NOT_RELEVANT = -1
    OTHER = 0
    BEGIN = 1
    INSIDE = 2


class GTSMatrixID(Enum):
    NOT_RELEVANT = -1
    OTHER = 0
    TARGET = 1
    OPINION = 2
    NEGATIVE = 3 if config['task']['type'] == 'triplet' else 3
    NEUTRAL = 4 if config['task']['type'] == 'triplet' else 3
    POSITIVE = 5 if config['task']['type'] == 'triplet' else 3


class MaskID(Enum):
    RELEVANT = 1
    NOT_RELEVANT = 0
