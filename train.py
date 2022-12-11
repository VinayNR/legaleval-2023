from flair.data import Corpus
from flair.datasets import ColumnCorpus

from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

# defining columns
columns = {0: 'text', 1: 'ner'}

# reading custom training data
data_folder = 'data'

# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(data_folder, columns, train_file='train.txt')

# print length of corpus
len(corpus.train)
len(corpus.dev)
len(corpus.test)

# print corpus train sample
print(corpus.train[0])

# custom training the model with the corpus
label_type = 'ner'

# make the label dictionary from the corpus
label_dict = corpus.make_label_dictionary(label_type=label_type)
print(label_dict)

# initialize embedding stack with Flair and GloVe
embedding_types = [
    WordEmbeddings('glove'),
    FlairEmbeddings('news-forward-fast'),
    FlairEmbeddings('news-backward-fast'),
]

embeddings = StackedEmbeddings(embeddings=embedding_types)

# initialize sequence tagger
tagger = SequenceTagger(hidden_size=128,
                        embeddings=embeddings,
                        tag_dictionary=label_dict,
                        tag_type=label_type,
                        use_crf=True)

# initialize trainer
trainer = ModelTrainer(tagger, corpus)

print('Before training')

# start training
trainer.train('resources/taggers/legal-ner',
              learning_rate=0.05,
              mini_batch_size=32,
              max_epochs=2)

print('After training')