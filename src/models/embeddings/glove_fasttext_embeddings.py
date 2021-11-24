import numpy as np
from tensorflow import keras
from typing import Dict, List
import tensorflow as tf
import fasttext
import sys
import os
import json
from src.config_reader import config


class GloveFasttext(keras.layers.Layer):
    def __init__(self):
        super(GloveFasttext, self).__init__()
        fasttext_path: str = os.path.join(os.path.dirname(sys.modules['__main__'].__file__), 'datasets',
                                          'datasets', 'embeddings_data',
                                          config['encoder']['glove-fasttext']['fasttext-model'])
        glove_path: str = os.path.join(os.path.dirname(sys.modules['__main__'].__file__), 'datasets',
                                       'datasets', 'embeddings_data', config['encoder']['glove-fasttext']['glove-file'])
        indexer_path: str = os.path.join(os.path.dirname(sys.modules['__main__'].__file__), 'datasets',
                                         'datasets', 'embeddings_data',
                                         config['encoder']['glove-fasttext']['indexer-path'])

        self.fasttext_model: fasttext = fasttext.load_model(fasttext_path)
        self.glove: np.ndarray = np.load(glove_path)
        self.word_to_idx: Dict = json.load(open(indexer_path))

    def call(self, inputs: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        out: np.ndarray = np.empty(shape=(*inputs.shape, config['encoder']['embedding_dimension']), dtype=np.float32)
        batch_idx: int
        for batch_idx, sentence in enumerate(inputs):
            word_idx: int
            word: tf.Tensor
            for word_idx, word in enumerate(sentence):
                word: str = bytes.decode(word.numpy())
                fasttext_temp: np.ndarray = self.fasttext_model.get_word_vector(word)
                glove_temp: np.ndarray = self.glove[self.word_to_idx[word]]
                out[batch_idx][word_idx] = np.concatenate((glove_temp, fasttext_temp), axis=-1)
        return tf.constant(np.expand_dims(mask, axis=-1) * out)
