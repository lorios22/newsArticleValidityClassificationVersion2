from typing import List, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_top_contributing_words(
    text: str,
    shap_values: np.ndarray,
    tokenizer: tf.keras.preprocessing.text.Tokenizer,
    n: int = 5
) -> List[Tuple[str, float]]:
    """
    Get the top N contributing words to the SHAP values.

    Args:
        text (str): The input text for which to calculate word importance.
        shap_values (np.ndarray): The SHAP values corresponding to the words in the text.
        tokenizer (tf.keras.preprocessing.text.Tokenizer): The tokenizer used to encode the text.
        n (int, optional): The number of top contributing words to return. Default is 5.

    Returns:
        List[Tuple[str, float]]: A list of tuples where each tuple contains a word and its associated SHAP value,
                                  sorted by the SHAP value in descending order.
    """
    words = text.split()
    word_shap = []
    for word in words:
        word_id = tokenizer.word_index.get(word.lower(), 0)
        if word_id != 0 and word_id < len(shap_values):
            word_shap.append((word, abs(shap_values[word_id])))
    return sorted(word_shap, key=lambda x: x[1], reverse=True)[:n]

def generate_explanation(
    text: str,
    prediction: float,
    shap_values: np.ndarray,
    tokenizer: tf.keras.preprocessing.text.Tokenizer
) -> str:
    """
    Generate a textual explanation of the model's prediction and top contributing words.

    Args:
        text (str): The input text that was classified.
        prediction (float): The predicted probability of the text being classified as valid.
        shap_values (np.ndarray): The SHAP values corresponding to the words in the text.
        tokenizer (tf.keras.preprocessing.text.Tokenizer): The tokenizer used to encode the text.

    Returns:
        str: A textual explanation of the prediction, including the top contributing words and their SHAP values.
    """
    label = 'Valid' if prediction > 0.5 else 'Invalid'
    top_words = get_top_contributing_words(text, shap_values, tokenizer)
    
    explanation = f"This article was classified as {label.lower()} (probability: {prediction:.4f}). "
    explanation += "The main contributing words were: "
    explanation += ", ".join([f"{word} ({shap:.4f})" for word, shap in top_words])
    
    return explanation

def word_importance(text, model, tokenizer):
    """
    Calculate word importance by analyzing changes in prediction.

    Args:
        text (str): The input text for which to compute word importance.
        model (tf.keras.Model): Trained Keras model.
        tokenizer (Tokenizer): Tokenizer used for text encoding.

    Returns:
        list of tuples: List of tuples containing words and their importance values, sorted by importance.
    """
    words = text.split()
    # Predict the base prediction for the original text
    base_prediction = model.predict(pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=200))[0][0]
    importances = []
    
    for i, word in enumerate(words):
        # Replace the current word with '<UNK>'
        new_text = ' '.join(words[:i] + ['<UNK>'] + words[i+1:])
        # Predict the new prediction
        new_prediction = model.predict(pad_sequences(tokenizer.texts_to_sequences([new_text]), maxlen=200))[0][0]
        # Calculate importance
        importance = abs(base_prediction - new_prediction)
        importances.append((word, importance))
    
    return sorted(importances, key=lambda x: x[1], reverse=True)
