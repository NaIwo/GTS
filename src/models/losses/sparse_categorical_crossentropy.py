import tensorflow as tf


class SparseCategoricalCrossentropy(tf.keras.losses.Loss):

    def __init__(self, **kwargs):
        super(SparseCategoricalCrossentropy, self).__init__(**kwargs)
        self.loss: tf.keras.losses = tf.keras.losses.CategoricalCrossentropy(from_logits=False,
                                                                             reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor, ignore_index: int = None, **kwargs) -> tf.Tensor:
        y_pred = tf.math.argmax(y_pred, axis=-1)
        if ignore_index is None:
            return tf.math.reduce_sum(self.loss(y_true, y_pred), axis=0)
        else:
            y_pred = tf.where(y_true == ignore_index, y_true, y_pred)