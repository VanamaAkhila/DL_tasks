
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Decoding the first review
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print(f"Decoded Review:\n{decoded_review}")

# One-hot encoding of words
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts([decoded_review])
encoded_words = tokenizer.texts_to_matrix([decoded_review], mode='binary')

print(f"One-hot encoded words (binary format):\n{encoded_words}")

# Printing one-hot encoding of words
words = decoded_review.split()
print("\nOne-hot encoding of words:")
for word, encoding in zip(words, encoded_words[0]):
    print(f"Word: '{word}' => One-hot encoding: {encoding}")

# Function to one-hot encode characters
def one_hot_encode_characters(text_data):
    unique_chars = sorted(set(''.join(text_data)))
    char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
    print(f"Character to index mapping: {char_to_index}")

    one_hot_vectors = []
    for text in text_data:
        vector = []
        for char in text:
            one_hot = [0] * len(unique_chars)
            one_hot[char_to_index[char]] = 1
            vector.append(one_hot)
        one_hot_vectors.append(vector)

    return one_hot_vectors

# One-hot encoding characters in the decoded review
character_encoded = one_hot_encode_characters([decoded_review])

# Printing character one-hot encodings
for idx, (char, encoding) in enumerate(zip(decoded_review, character_encoded[0])):
    print(f"Character: '{char}' => One-hot encoding: {encoding}")
