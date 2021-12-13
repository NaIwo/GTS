from src.utils import config

from tensorflow import keras
import tensorflow as tf


class Inference(keras.layers.Layer):
    def __init__(self):
        super(Inference, self).__init__()
        model_type: str = config['model']['type']
        self.inference_quantity: int = config['model'][model_type]['inference']  # L hyperparameter in article

        encoder_type: str = config['model']['type']
        if model_type == 'bert':
            units: int = config['encoder'][encoder_type]['embedding-dimension'] * 2
        elif model_type == 'cnn':
            units: int = 256 * 2  # 256 - cnn output dimension - paper
        else:  # bilstm
            units: int = 100 * 2  # 100 - bilstm output dimension (50 * 2 - concatenation) - paper
        self.linear: keras.layers.Layer = tf.keras.layers.Dense(units, activation='linear')

        classes: int = config['task']['class-number'][config['task']['type']]
        self.softmax: keras.layers.Layer = tf.keras.layers.Dense(classes, activation='softmax')

    def call(self, embeddings: tf.Tensor, mask: tf.Tensor, **kwargs) -> tf.Tensor:
        matrix: tf.Tensor = self._get_embeddings_matrix(embeddings=embeddings)

        p_ij: tf.Tensor = self.softmax(matrix)
        z_ij: tf.Tensor = matrix

        t: int
        for t in range(self.inference_quantity):
            p_maxpool: tf.Tensor = self._maxpool_phase(p_ij * mask)
            q_ij: tf.Tensor = tf.concat([z_ij, p_maxpool, p_ij], axis=3)
            z_ij = self.linear(q_ij)
            p_ij = self.softmax(z_ij)
        return p_ij

    @staticmethod
    def _get_embeddings_matrix(embeddings: tf.Tensor) -> tf.Tensor:
        expanded_embeddings: tf.Tensor = tf.expand_dims(embeddings, axis=2)

        emb: tf.Tensor = tf.repeat(expanded_embeddings, repeats=expanded_embeddings.shape[1], axis=2)
        emb_t: tf.Tensor = tf.transpose(emb, perm=[0, 2, 1, 3])

        return tf.concat([emb, emb_t], axis=-1)

    @staticmethod
    def _maxpool_phase(p_ij: tf.Tensor) -> tf.Tensor:
        p_i: tf.Tensor = tf.math.reduce_max(p_ij, axis=1)
        p_i = tf.expand_dims(p_i, axis=3)

        p_j: tf.Tensor = tf.math.reduce_max(p_ij, axis=2)
        p_j = tf.expand_dims(p_j, axis=3)

        p_concat: tf.Tensor = tf.concat([p_i, p_j], axis=3)
        p_max: tf.Tensor = tf.math.reduce_max(p_concat, axis=3)
        p_max = tf.expand_dims(p_max, axis=2)
        p_max = tf.repeat(p_max, repeats=p_max.shape[1], axis=2)
        p_t: tf.Tensor = tf.transpose(p_max, perm=[0, 2, 1, 3])

        return tf.concat([p_max, p_t], axis=3)
