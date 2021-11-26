from datasets import DatasetReader, Dataset

from config_reader import config
from models import BaseModel, get_model

if __name__ == "__main__":
    dataset_reader: DatasetReader = DatasetReader(dataset_name=config['dataset']['dataset-name'])

    train_ds: Dataset = dataset_reader.read('train')
    dev_ds: Dataset = dataset_reader.read('dev')
    test_ds: Dataset = dataset_reader.read('test')

    model: BaseModel = get_model()

    for data in train_ds.batch(batch_size=config['dataset']['batch-size'], seed=config['dataset']['shuffle-seed']):
        print(model(data))
        break
