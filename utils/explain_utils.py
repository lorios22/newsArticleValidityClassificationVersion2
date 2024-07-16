from typing import List, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_top_contributing_words(text: str, shap_values: np.ndarray, tokenizer: tf.keras.preprocessing.text.Tokenizer, n: int = 5) -> List[Tuple[str, float]]:
    words = text.split()
    word_shap = []
    for word in words:
        word_id = tokenizer.word_index.get(word.lower(), 0)
        if word_id != 0 and word_id < len(shap_values):
            word_shap.append((word, abs(shap_values[word_id])))
    return sorted(word_shap, key=lambda x: x[1], reverse=True)[:n]

def generate_explanation(text: str, prediction: float, shap_values: np.ndarray, tokenizer: tf.keras.preprocessing.text.Tokenizer) -> str:
    label = 'Valid' if prediction > 0.5 else 'Invalid'
    top_words = get_top_contributing_words(text, shap_values, tokenizer)
    
    explanation = f"This article was classified as {label.lower()} (probability: {prediction:.4f}). "
    explanation += "The main contributing words were: "
    explanation += ", ".join([f"{word} ({shap:.4f})" for word, shap in top_words])
    
    return explanation

def word_importance(text, model, tokenizer, MAX_LEN):
    """Calculate word importance by analyzing changes in prediction."""
    words = text.split()
    base_prediction = model.predict(pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=MAX_LEN))[0][0]
    importances = []
    
    for i, word in enumerate(words):
        new_text = ' '.join(words[:i] + ['<UNK>'] + words[i+1:])
        new_prediction = model.predict(pad_sequences(tokenizer.texts_to_sequences([new_text]), maxlen=MAX_LEN))[0][0]
        importance = abs(base_prediction - new_prediction)
        importances.append((word, importance))
    
    return sorted(importances, key=lambda x: x[1], reverse=True)