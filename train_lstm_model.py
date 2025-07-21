import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load dataset (make sure 'features.xlsx' is in your directory)
df = pd.read_excel('features.xlsx')

texts = df['processed_text'].astype(str).tolist()   # Input
labels = df['label'].astype(str).tolist()           # Output

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
y = to_categorical(encoded_labels)

# Tokenize text
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=200)

# Build LSTM model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=64, input_length=200))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(padded_sequences, y, epochs=5, batch_size=32, validation_split=0.2)

# Save model and encoders
model.save('model/lstm_model.h5')

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Model trained and files saved: lstm_model.h5, tokenizer.pkl, label_encoder.pkl")
