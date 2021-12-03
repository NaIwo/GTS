import tensorflow as tf


def trim1d(matrix: tf.Tensor, length: int, offset: int = 0) -> tf.Tensor:
    return matrix[:, offset:offset + length]


def trim2d(matrix: tf.Tensor, length: int, offset: int = 0) -> tf.Tensor:
    return matrix[:, offset:offset + length, offset:offset + length]
