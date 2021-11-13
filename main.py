from datasets import DatasetReader

if __name__ == "__main__":
    dataset = DatasetReader(dataset_name='lap14')
    for data in dataset.train_dataset:
        pass