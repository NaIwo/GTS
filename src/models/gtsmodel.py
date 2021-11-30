from .models_instances import BaseModel, BertModel
from src.config_reader import config
from src.datasets import Dataset
from src.datasets.domain import IgnoreIndex
from .losses.sparse_categorical_crossentropy import SparseCategoricalCrossentropy
from .metrics.gts_metric import GtsMetric

import tensorflow as tf
from typing import List


class GtsModel:
    def __init__(self, model: BaseModel):
        self._model: BaseModel = model

    def __call__(self, data: Dataset, training: bool = False, **kwargs) -> tf.Tensor:
        return self._model(inputs=data, training=training, **kwargs)

    def train(self, train_data: Dataset, dev_data: Dataset, **kwargs):
        loss_fn = self._get_loss_function()
        optimizer: tf.keras.optimizers = self._get_optimizer()

        epoch: int
        epochs: int = config['model']['epochs']
        for epoch in range(epochs):
            metrics: List = self._get_metrics()
            step: int
            for step, data in enumerate(train_data):
                with tf.GradientTape() as tape:
                    prediction: tf.Tensor = self(data=data, training=True, **kwargs)
                    loss_value: tf.Tensor = loss_fn(y_true=data.gts_matrix, y_pred=prediction)
                grads = tape.gradient(loss_value, self._model.trainable_weights)
                optimizer.apply_gradients(zip(grads, self._model.trainable_weights))
                self._update_metrics(metrics=metrics, y_true=data, y_pred=prediction)

                print(f"\rEpoch {epoch + 1}/{epochs}, Step {step}, Loss {loss_value}, Metric: {metrics[0].result()}", end='', flush=True)

    @staticmethod
    def _get_loss_function() -> SparseCategoricalCrossentropy:
        return SparseCategoricalCrossentropy(ignore_index=IgnoreIndex.IGNORE_INDEX.value, from_logits=False)

    @staticmethod
    def _get_optimizer() -> tf.keras.optimizers:
        return tf.keras.optimizers.Adam(config['model']['learning-rate'])

    @staticmethod
    def _get_metrics() -> List:
        return [GtsMetric()]

    @staticmethod
    def _update_metrics(metrics: List, y_true: Dataset, y_pred: tf.Tensor) -> None:
        for metric in metrics:
            metric.update_state(y_true=y_true, y_pred=y_pred)


if config['encoder']['type'] == 'bert':
    gts_model: GtsModel = GtsModel(BertModel())
elif config['encoder']['type'] == 'cnn':
    gts_model: GtsModel = GtsModel(BertModel())
elif config['encoder']['type'] == 'bilstm':
    gts_model: GtsModel = GtsModel(BertModel())
