from tensorflow import keras
import tensorflow as tf
from transformers import TFBertModel

from src.config_reader import config


class Bert(keras.layers.Layer):
    def __init__(self):
        super(Bert, self).__init__()
        self.bert: TFBertModel = TFBertModel.from_pretrained(config['encoder']['bert']['source'])

    def call(self, inputs: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        return self.bert([inputs, mask]).last_hidden_state
