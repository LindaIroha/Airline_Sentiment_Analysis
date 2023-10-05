import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# Load the trained model
model = load_model('model.h5')

# Load the saved tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

#Define the preprocess text function
def preprocess_text(text):
    #Tokenize the input text
    tokens = tokenizer.texts_to_sequences([text])

    # pad the sequences to a fixed length (use the same sequence length as during training)
    #Max_sequence_length = ... # Replace with the sequence length used during training

    padded_tokens = pad_sequences(tokens, maxlen = 100)

    return padded_tokens[0]


# Create the title for the App
st.title ('Airline Sentiment Analysis App')

# Create a text input widget for user input
user_input = st.text_area('Enter text for Airline Sentiment Analysis', '')

# Create a button to trigger sentiment analysis
if st.button('Analyze Sentiment'):
    # Preprocess the user input
    processed_input = preprocess_text(user_input)

    # Make predictions using the loaded model
    prediction = model.predict(np.array([processed_input]))
    st.write(prediction)

    if prediction[0][0] > 0.1:
        sentiment = 'Negative'
    elif prediction[0][1] > 0.1:
        sentiment = 'Neutral'
    else:
        sentiment = 'Positive'

 
    #Dispaly the Sentiment
    st.write(f' ### Sentiment: {sentiment}')


