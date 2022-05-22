""" Word Tokenizer
Before start of any NLP related work, text must be converted to number.
Each sentences become a sequences of number.

- Tokenizer class provide method to create sequences
  1. First create Tokenizer class
  2. Fit the dataset to the tokenizer object
  3. Finally, use text_to_sequence to convert all text to sequences

- pad_sequences provide method to get similar length sequences for all sentences
"""

# %%
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# %%
# Example 1
sentences = ['i love my dog', 'I, love my cat']
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)

print(tokenizer.word_index)
print(tokenizer.index_word)

# %%
# Example 2
sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?',
]
tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

print(tokenizer.word_index)
print(sequences)

# %%
# Example 3
test_data = [
    'i really love my dog',
    'my dog loves my manatee',
]
print(tokenizer.texts_to_sequences(test_data))

# %%
# Example for pad_sequences
padded = pad_sequences(sequences)
print(padded)
print(pad_sequences(sequences, maxlen=10, padding='post', truncating='post'))
