from enum import Enum
from src.utils import config


class IgnoreIndex(Enum):
    IGNORE_INDEX = -1


class TagVectorID(Enum):
    NOT_RELEVANT = IgnoreIndex.IGNORE_INDEX.value
    OTHER = 0
    BEGIN = 1
    INSIDE = 2


class GTSMatrixID(Enum):
    NOT_RELEVANT = IgnoreIndex.IGNORE_INDEX.value
    OTHER = 0
    TARGET = 1
    OPINION = 2
    PAIR = 3
    NEGATIVE = 3
    NEUTRAL = 4 if config['task']['type'] == 'triplet' else 3
    POSITIVE = 5 if config['task']['type'] == 'triplet' else 3


class MaskID(Enum):
    NOT_RELEVANT = 0
    RELEVANT = 1
