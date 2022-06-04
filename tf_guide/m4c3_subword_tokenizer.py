"""
uses of sub word tokenizer
"""
# %%

from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
import tensorflow_datasets as tfds

# %% Download the datasets
# plain text default config
imdb_plaintext, info_plaintext = tfds.load('imdb_reviews',
                                           with_info=True,
                                           as_supervised=True)

# sub word encoded pre tokenized dataset
imdb_sub_words, info_sub_words = tfds.load('imdb_reviews/subwords8k',
                                           with_info=True,
                                           as_supervised=True)

# %% check difference between two dataset
print(info_plaintext.features)
for example in imdb_plaintext['train'].take(2):
  print(example[0].numpy())

print(info_sub_words.features)
for example in imdb_sub_words['train'].take(2):
  print(example[0].numpy())

# %% use decode to view the text feature
tokenizer_sub_words = info_sub_words.features['text'].encoder
for example in imdb_sub_words['train'].take(2):
  print(tokenizer_sub_words.decode(example[0]))

# %% Use sample string to understand how sub word tokenizer work
sample_string = 'TensorFlow, from basics to mastery'

# Encode and print the results
tokenized_string = tokenizer_sub_words.encode(sample_string)
print(f'Tokenized string is {tokenized_string}')

# Decode and print the results
original_string = tokenizer_sub_words.decode(tokenized_string)
print(f'The original string: {original_string}')

# Print each token with sub word value
for ts in tokenized_string:
  print(f'{ts} ----> {tokenizer_sub_words.decode([ts])}')

# %% Prepare the data
BUFFER_SIZE = 10000
BATCH_SIZE = 64

# Get the train and test splits
train_data, test_data = imdb_sub_words['train'], imdb_sub_words['test']
train_dataset = train_data.shuffle(BUFFER_SIZE)

# Batch and pad the datasets to the maximum length of the sequences
train_dataset = train_dataset.padded_batch(BATCH_SIZE)
test_dataset = test_data.padded_batch(BATCH_SIZE)

# %% Build and Train the model
clear_session()
model = Sequential([
    Embedding(tokenizer_sub_words.vocab_size, 64),
    GlobalAveragePooling1D(),
    Dense(6, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()
model.fit(train_dataset, epochs=10, validation_data=test_dataset)
