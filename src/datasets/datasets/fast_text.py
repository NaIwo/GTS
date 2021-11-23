import os
import json
from typing import Dict, List
import fasttext
import argparse


def train_fasttext(dataset_name):
    out_path: str = os.path.join('embeddings_data', f'model_{dataset_name}.bin')
    output_sentences: str = ''
    temp_file_name: str = 'train.txt'
    filename: str
    for filename in ['train.json', 'dev.json']:
        with open(os.path.join(dataset_name, filename), 'r') as file:
            sentences: List = json.load(file)

        sentence: Dict
        for sentence in sentences:
            output_sentences += sentence['sentence'] + ' '
    with open(temp_file_name, 'w') as file:
        file.write(output_sentences)

    model: fasttext = fasttext.train_unsupervised('train.txt', epoch=30, dim=100)
    os.remove(temp_file_name)
    model.save_model(out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, choices=['lap14', 'res14', 'res15', 'res16'], required=True,
                        help='Dataset name to train')

    args = parser.parse_args()
    train_fasttext(dataset_name=args.name)
