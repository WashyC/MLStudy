""" Binary classification on Text Data
- Embedding layer.
  The main idea here is to represent each word in the vocabulary with vectors.
  These vectors have trainable weights so as your neural network learns, words
  that are most likely to appear in a positive tweet will converge towards
  similar weights. Similarly, words in negative tweets will be clustered more
  closely together. You can read more about word embeddings
  https://www.tensorflow.org/text/guide/word_embeddings
"""

# %%

import numpy as np
import tensorflow_datasets as tfds

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# %%
# Load dataset
imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
# check dataset samples
print(info)
for rev in imdb['train'].take(2):
  print(rev)

# %%
# Prepare the dataset
train_sentences, train_labels = [], []
for review, label in imdb['train']:
  train_sentences.append(review.numpy().decode('utf8'))
  train_labels.append(label.numpy())
train_labels = np.array(train_labels)

test_sentences, test_labels = [], []
for review, label in imdb['test']:
  test_sentences.append(review.numpy().decode('utf8'))
  test_labels.append(label.numpy())
test_labels = np.array(test_labels)

# %%
# Tokenize the data
vocab_size = 10000
max_length = 120

tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(train_sentences)

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences,
                             maxlen=max_length,
                             truncating='post')

test_sentences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sentences,
                            maxlen=max_length,
                            truncating='post')

# %%
# Build and train the model
model = Sequential([
    Embedding(vocab_size, 16, input_length=max_length),
    Flatten(),
    Dense(6, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()
model.fit(train_padded,
          train_labels,
          epochs=10,
          validation_data=(test_padded, test_labels))

# %%
# Check the word features
index_word = tokenizer.index_word
embedding_weight = model.layers[0].get_weights()[0]
print(embedding_weight.shape)

for i in range(1, 10):
  print(index_word[i], embedding_weight[i])

# %%
# create visualization
# https://projector.tensorflow.org/
with open('meta.tsv', 'w', encoding='utf-8') as out_m, \
  open('vectors.tsv', 'w', encoding='utf-8') as out_v:
  for idx in range(1, vocab_size):
    out_m.write(index_word[idx] + '\n')
    out_v.write('\t'.join([str(x) for x in embedding_weight[idx]]) + '\n')
