from .models_instances import BaseModel, BertModel, CnnModel, BilstmModel
from src.utils import config
from src.datasets import Dataset
from src.datasets.domain import IgnoreIndex
from .losses.sparse_categorical_crossentropy import SparseCategoricalCrossentropy
from .metrics.gts_metric import GtsMetric

import tensorflow as tf
from typing import Dict, Optional, DefaultDict
from collections import defaultdict
import yaml
import os
import logging
from datetime import datetime


class Memory:
    def __init__(self):
        self.best_epoch: int = 0
        self.best_score: float = 0.0
        self.current_epoch: int = 0
        self._losses: DefaultDict = defaultdict(list)

    def is_best(self, metrics: Dict) -> bool:
        metric = metrics['GtsMetric']
        score: float = metric.result()['Union score']['f1']
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = self.current_epoch
            return True
        else:
            return False

    def update_loss(self, loss: tf.Tensor) -> None:
        self._losses[self.current_epoch].append(loss)

    @property
    def loss(self) -> float:
        return float(tf.math.reduce_mean(self._losses[self.current_epoch]))


class GtsModel:
    def __init__(self, model: BaseModel, model_type: str):
        self._model: BaseModel = model
        self.model_type: str = model_type
        self.memory: Optional[Memory] = None

    def __call__(self, training: bool = False, **kwargs) -> tf.Tensor:
        return self._model(training=training, **kwargs)

    def train(self, train_data: Dataset, dev_data: Dataset, **kwargs):
        logging.info(f'Start Training: {datetime.now()}')

        self.memory = Memory()
        loss_fn = self._get_loss_function()
        optimizer: tf.keras.optimizers = self._get_optimizer()
        epoch: int
        epochs: int = config['model'][self.model_type]['epochs']
        for epoch in range(epochs):

            logging.info(f'Epoch: {epoch}')
            self.memory.current_epoch = epoch

            step: int
            for step, data in enumerate(train_data):
                with tf.GradientTape() as tape:
                    prediction: tf.Tensor = self(**self.get_input_data(data), training=True, **kwargs)
                    loss_value: tf.Tensor = loss_fn(y_true=tf.convert_to_tensor(data.gts_matrix), y_pred=prediction)
                grads = tape.gradient(loss_value, self._model.trainable_weights)
                optimizer.apply_gradients([
                    (grad, var)
                    for (grad, var) in zip(grads, self._model.trainable_weights)
                    if grad is not None
                ])
                self.memory.update_loss(loss_value)

                print(f"\rEpoch {epoch + 1}/{epochs}, Step {step}, Loss {loss_value}", end='', flush=True)

                self.test(test_data=dev_data.sample_data(config['dataset']['valid-sample-ratio']))
            logging.info(f'Average epoch loss ({epoch}): {self.memory.loss}')

        logging.info(f'End Training: {datetime.now()}')
        logging.info(f'Best epoch: {self.memory.best_epoch}')
        self.memory = None

    def test(self, test_data: Dataset, **kwargs) -> None:
        metrics: Dict = self._get_metrics()
        step: int
        for step, data in enumerate(test_data):
            prediction: tf.Tensor = self(**self.get_input_data(data), training=False, **kwargs)
            self._update_metrics(metrics=metrics, y_true=data, y_pred=prediction)
        self._print_metrics(metrics=metrics)

        self._save_weights_if_necessary(metrics)

        self._reset_metrics_state(metrics=metrics)

    @staticmethod
    def get_input_data(data: Dataset) -> Dict:
        return {
            'encoded_sentence': data.encoded_sentence,
            'mask': data.mask,
            'mask3d': data.mask3d,
            'sentence_length': data.sentence_length
        }

    def build(self, example_input_shape: tf.Tensor):
        self._model.build(example_input_shape)

    @staticmethod
    def _get_loss_function() -> SparseCategoricalCrossentropy:
        return SparseCategoricalCrossentropy(ignore_index=IgnoreIndex.IGNORE_INDEX.value)

    def _get_optimizer(self) -> tf.keras.optimizers:
        return tf.keras.optimizers.Adam(config['model'][self.model_type]['learning-rate'])

    @staticmethod
    def _get_metrics() -> Dict:
        return {'GtsMetric': GtsMetric()}

    @staticmethod
    def _update_metrics(metrics: Dict, y_true: Dataset, y_pred: tf.Tensor) -> None:
        for metric in metrics.values():
            metric.update_state(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _print_metrics(metrics: Dict, source: str = 'test') -> None:
        logging_str: str = ''
        logging_str += f'\n{source} metrics:'
        metric_name: str
        for metric_name, metric in metrics.items():
            logging_str += f'\n{metric_name}'
            logging_str += f'\n{yaml.dump(metric.result(), sort_keys=False, default_flow_style=False)}'
        logging.info(logging_str)

    @staticmethod
    def _reset_metrics_state(metrics: Dict) -> None:
        for metric in metrics.values():
            metric.reset_state()

    def _save_weights_if_necessary(self, metrics: Dict) -> None:
        if self.memory and (self.memory.is_best(metrics) or self.memory.current_epoch == 0):
            logging.info(f'Saving model weights to file: {self._get_file_path()}')
            self.save_weights()

    def save_weights(self) -> None:
        self._model.save_weights(self._get_file_path())

    def load_weights(self, file_path: Optional[str] = None) -> None:
        if not file_path:
            file_path = self._get_file_path()
        self._model.load_weights(file_path)

    def _get_file_path(self) -> str:
        file_name: str = f"{config['encoder']['type']}_{config['task']['type']}_{config['dataset']['batch-size']}"
        return os.path.join(os.getcwd(), 'models_weights', config['dataset']['dataset-name'], self.model_type, file_name)


if config['model']['type'] == 'bert':
    gts_model: GtsModel = GtsModel(BertModel(), 'bert')
elif config['model']['type'] == 'cnn':
    gts_model: GtsModel = GtsModel(CnnModel(), 'cnn')
elif config['model']['type'] == 'bilstm':
    gts_model: GtsModel = GtsModel(BilstmModel(), 'bilstm')
