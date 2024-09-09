import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import List, Tuple

def load_model_and_tokenizer(model_path, tokenizer_path):
    """
    Load a trained Keras model and tokenizer.

    Args:
        model_path (str): Path to the saved Keras model.
        tokenizer_path (str): Path to the saved tokenizer JSON file.

    Returns:
        tuple: A tuple containing:
            - model (tf.keras.Model): Loaded Keras model.
            - tokenizer (Tokenizer): Loaded tokenizer.
    """
    model = tf.keras.models.load_model(model_path)
    
    with open(tokenizer_path, 'r') as file:
        tokenizer_json = json.load(file)
        tokenizer = tokenizer_from_json(tokenizer_json)
    
    return model, tokenizer

def preprocess_text(texts, tokenizer, max_len=200):
    """
    Preprocess and pad text data for model prediction.

    Args:
        texts (list of str): List of text samples to preprocess.
        tokenizer (Tokenizer): Tokenizer used for text encoding.
        max_len (int, optional): Maximum length of the sequences. Default is 200.

    Returns:
        numpy array: Padded sequences ready for model prediction.
    """
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded_sequences

def predict_sentiment(text, model, tokenizer, max_len=200):
    """
    Make predictions with the model.

    Args:
        text (str): Input text for prediction.
        model (tf.keras.Model): Trained Keras model.
        tokenizer (Tokenizer): Tokenizer used for text encoding.
        max_len (int, optional): Maximum length of the sequences. Default is 200.

    Returns:
        tuple: A tuple containing:
            - sentiment (str): 'Valid' or 'Invalid' based on prediction.
            - confidence (float): Prediction probability.
    """
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    prediction = model.predict(padded)[0][0]
    return "Valid" if prediction > 0.5 else "Invalid", prediction

# Define the model prediction function for SHAP
def model_predict(model, X):
    """
    Make predictions with the model.

    Args:
        model (tf.keras.Model): Trained Keras model.
        X (numpy array): Input data.

    Returns:
        numpy array: Model predictions.
    """
    return model.predict(X)