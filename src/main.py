from datasets import DatasetReader, Dataset
from config_reader import config
from models import GtsModel, gts_model

if __name__ == "__main__":
    dataset_reader: DatasetReader = DatasetReader(dataset_name=config['dataset']['dataset-name'])

    train_ds: Dataset = dataset_reader.read('train')
    dev_ds: Dataset = dataset_reader.read('dev')
    test_ds: Dataset = dataset_reader.read('test')

    gts_model: GtsModel = gts_model

    gts_model.train(train_data=train_ds, dev_data=dev_ds)
