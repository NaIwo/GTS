import tensorflow as tf

from ..utils import trim2d


class SparseCategoricalCrossentropy(tf.keras.losses.Loss):

    def __init__(self, ignore_index: int = None, **kwargs):
        super(SparseCategoricalCrossentropy, self).__init__(**kwargs)
        self.ignore_index: int = ignore_index

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = trim2d(y_true, y_pred.shape[1])

        mask: tf.Tensor = tf.where(y_true == self.ignore_index, 0.0, 1.0)

        y_true = tf.where(y_true == self.ignore_index, 0, y_true)

        probabilities: tf.Tensor = self.gather(y_pred, y_true)
        log_probabilities: tf.Tensor = tf.math.log(probabilities)
        log_probabilities *= mask
        batch_loss: tf.Tensor = tf.math.reduce_sum(log_probabilities, axis=(1, 2))
        normalizer: tf.Tensor = tf.math.reduce_sum(mask)
        return -tf.math.reduce_sum(batch_loss) / normalizer

    @staticmethod
    def gather(params: tf.Tensor, indices: tf.Tensor) -> tf.Tensor:
        dim: int
        indexes: tf.Tensor = tf.meshgrid(*[tf.range(dim, dtype=tf.int64) for dim in indices.shape], indexing='ij')
        indexes = tf.transpose(tf.stack(indexes), perm=[1, 2, 3, 0])
        indexes = tf.concat([indexes, tf.expand_dims(indices, axis=-1)], axis=-1)
        return tf.gather_nd(params, indexes)
