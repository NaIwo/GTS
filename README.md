# GTS - Grid Tagging Scheme

Tensorflow implementation of Grid Tagging Scheme for Aspect-oriented Fine-grained Opinion Extraction. 

Original paper: https://aclanthology.org/2020.findings-emnlp.234.pdf

Authors' code: https://github.com/NJUNLP/GTS

## Requirements
Tested with:

python == 3.8

tensorflow == 2.6.0

All of required packages are provided in **requirements.txt** file.

Just run:
```
pip install -r requirements.txt
```
## Usage
### Setting PythonPath
```
export PYTHONPATH="${PYTHONPATH}:/path/to/project/GTS"
```
### Run
```
python src/main.py
```
#### Parametrization
All params settings are included in **config.yml**. You can manipulate them to set up your own experiments.

Fields worth explaining in **config.yml**
- valid-sample-ratio - [0.0;1.0] - Ratio specifying the percentage of sampled data from the validation set. Validation is carried out on such a selected subset (calculated metrics) - this allows for faster calculation of metrics at the expense of a smaller set.
- task - {triplet, pair} - task type - pair is a task which cares only about pairing target with opinion without sentiment - like in paper.
- max-length - Maximum length of sentences - shorter ones will be padded.
- indexer - this encoder is using prepared embeddings from orginal repsoitory.
- glove-fasttext - this encoder is using your own fasttext embeddings - you can prepare fasttext embeddings by your own with  *src/datasets/datasets/fast_text.py* script.
#### Matrix fields marking
If you want to use different markings in the matrix, change the values in *src/datasets/domain/enums.py* the file to the ones you prefer.
