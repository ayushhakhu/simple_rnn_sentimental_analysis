# Step 1 : import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence


# Load IMDB word index
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in text])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3  for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500, padding='pre')
    return padded_review

# Load Model file
model = tf.keras.models.load_model('../simple_rnn_imdb.h5')


# step 3 : predict sentiment
def predict_sentiment(text):
    padded_review = preprocess_text(text)
    prediction = model.predict(padded_review)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    return sentiment, prediction[0][0]


# create streamlit app
import streamlit as st
st.title('Sentiment Analysis')

text = st.text_area('Enter a text to predict sentiment')

if st.button('Predict'):
    sentiment, score = predict_sentiment(text)
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Score: {score*100:.2f}%')

if __name__ == '__init__':
    main()

