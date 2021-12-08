from datasets import DatasetReader, Dataset
from utils import config, set_up_logger
from models import GtsModel, gts_model

import logging


def log_introductory_info() -> None:
    logging.info(f"Dataset: {config['dataset']['dataset-name']}")
    logging.info(f"Model type: {config['model']['type']}")
    logging.info(f"Encoder type: {config['encoder']['type']}")
    logging.info(f"Task type: {config['task']['type']}")
    logging.info(f"Batch size: {config['dataset']['batch-size']}")


if __name__ == "__main__":
    set_up_logger()
    log_introductory_info()

    dataset_reader: DatasetReader = DatasetReader(dataset_name=config['dataset']['dataset-name'])

    train_ds: Dataset = dataset_reader.read('train')
    dev_ds: Dataset = dataset_reader.read('dev')
    test_ds: Dataset = dataset_reader.read('test')

    gts_model: GtsModel = gts_model

    if config['model']['training']:
        gts_model.train(train_data=train_ds, dev_data=dev_ds)

    logging.info(f'TEST SET')
    gts_model.load_weights()
    gts_model.test(test_data=test_ds)
