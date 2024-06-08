import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import nltk

# Download NLTK stop words
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Add custom repetitive meaningless words
custom_stop_words = {'felt', 'feeling', 'feel', 'i felt', 'it was', 'was'}

# Load the dataset
df = pd.read_csv('../data/enISEAR_appraisal.tsv', sep='\t')
texts = df['Sentence'].str.lower().str.replace('.', '', regex=True).str.replace(',', '', regex=True).tolist()

# Function to remove stop words and custom meaningless words
def remove_stop_words(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words and word not in custom_stop_words]
    return ' '.join(filtered_words)

# Apply the function to all sentences
texts = [remove_stop_words(text) for text in texts]

# Tokenize the text
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=100)

# Save the tokenizer
with open('../data/tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Train Word2Vec model
sentences = [text.split() for text in texts]
word2vec_model = Word2Vec(sentences, vector_size=300, window=5, min_count=1, workers=4)
word2vec_model.save("../data/word2vec.model")

# Create the embedding matrix
embedding_dim = 300
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]

# Save the embedding matrix
np.save('../data/embedding_matrix.npy', embedding_matrix)

# Save the processed data
np.save('../data/data.npy', data)

# Prepare and save labels
label_columns = ['Attention', 'Certainty', 'Effort', 'Pleasant', 'Responsibility', 'Control', 'Circumstance']
label_sets = ['a1', 'a2', 'a3']

for label in label_columns:
    for label_set in label_sets:
        column_name = f'{label}_{label_set}'
        labels = df[column_name].values
        np.save(f'../data/labels_{column_name}.npy', labels)
