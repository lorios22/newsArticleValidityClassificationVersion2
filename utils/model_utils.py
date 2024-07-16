import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import List, Tuple

def load_model_and_tokenizer(model_path: str, tokenizer_path: str) -> Tuple[tf.keras.Model, tf.keras.preprocessing.text.Tokenizer]:
    try:
        model = load_model(model_path)
        with open(tokenizer_path, 'r') as f:
            tokenizer_json = json.load(f)
            tokenizer = tokenizer_from_json(tokenizer_json)
        print("Model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model or tokenizer: {str(e)}")
        raise

def preprocess_text(texts: List[str], tokenizer: tf.keras.preprocessing.text.Tokenizer, maxlen: int) -> np.ndarray:
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=maxlen)

def model_predict(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    return model.predict(X)
