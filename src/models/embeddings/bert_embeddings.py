from tensorflow import keras
import tensorflow as tf
from transformers import TFBertModel, BertConfig

from src.config_reader import config


class Bert(keras.layers.Layer):
    def __init__(self):
        super(Bert, self).__init__()
        num_labels: int = config['task']['class-number'][config['task']['type']]
        bert_config = BertConfig.from_pretrained(config['encoder']['bert']['source'], num_labels=num_labels)
        self.bert: TFBertModel = TFBertModel.from_pretrained(config['encoder']['bert']['source'], config=bert_config)

    def call(self, inputs: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        return self.bert([inputs, mask]).last_hidden_state
