import numpy as np
from tensorflow import keras
import tensorflow as tf
import sys
import os
from src.config_reader import config


class Indexer(keras.layers.Layer):
    def __init__(self):
        super(Indexer, self).__init__()
        fasttext_path: str = os.path.join(os.path.dirname(sys.modules['__main__'].__file__), 'datasets',
                                          'datasets', 'embeddings_data',
                                          config['encoder']['indexer']['fasttext-file'])
        glove_path: str = os.path.join(os.path.dirname(sys.modules['__main__'].__file__), 'datasets',
                                       'datasets', 'embeddings_data', config['encoder']['indexer']['glove-file'])

        self.fasttext: np.ndarray = np.load(fasttext_path)
        self.glove: np.ndarray = np.load(glove_path)

    def call(self, inputs: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        out: np.ndarray = np.empty(shape=(*inputs.shape, config['encoder']['indexer']['embedding-dimension']),
                                   dtype=np.float32)
        batch_idx: int
        for batch_idx, sentence in enumerate(inputs):
            fasttext_temp: np.ndarray = self.fasttext[sentence]
            glove_temp: np.ndarray = self.glove[sentence]
            out[batch_idx] = np.concatenate((glove_temp, fasttext_temp), axis=-1)
        return tf.convert_to_tensor(np.expand_dims(mask, axis=-1) * out)
