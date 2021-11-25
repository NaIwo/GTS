from src.config_reader import config

from .glove_fasttext_embeddings import GloveFasttext
from .indexer_embeddings import Indexer
from .bert_embeddings import Bert

if config['encoder']['type'] == 'bert':
    embeddings_layer: Bert = Bert()
elif config['encoder']['type'] == 'indexer':
    embeddings_layer: Indexer = Indexer()
else:
    embeddings_layer: GloveFasttext = GloveFasttext()
