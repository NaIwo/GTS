import tensorflow as tf


class SparseCategoricalCrossentropy(tf.keras.losses.Loss):

    def __init__(self, ignore_index: int = None, from_logits: bool = True, **kwargs):
        super(SparseCategoricalCrossentropy, self).__init__(**kwargs)
        self.loss: tf.keras.losses = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits,
                                                                                   reduction=tf.keras.losses.Reduction.NONE)
        self.ignore_index: int = ignore_index

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        if self.ignore_index is None:
            return tf.math.reduce_sum(self.loss(y_true, y_pred))
        else:
            mask: tf.Tensor = tf.where(y_true == self.ignore_index, 0.0, 1.0)
            y_true = tf.where(y_true == self.ignore_index, 0, y_true)
            loss = self.loss(y_true, y_pred)
            return tf.math.reduce_sum(loss * mask)
