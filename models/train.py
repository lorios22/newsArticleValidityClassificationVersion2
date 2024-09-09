import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

def tokenize_and_pad(X_train, X_val, max_words=50000, max_len=200):
    """
    Tokenizes and pads the input text data for training and validation.

    Args:
        X_train (list): List of training text data.
        X_val (list): List of validation text data.
        max_words (int): Maximum number of words to keep based on word frequency. Default is 50,000.
        max_len (int): Maximum length of all sequences. Default is 200.

    Returns:
        tuple: Padded sequences for training and validation data, and the fitted tokenizer.
    """
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_val_pad = pad_sequences(X_val_seq, maxlen=max_len, padding='post', truncating='post')
    
    return X_train_pad, X_val_pad, tokenizer

def build_model(max_length, vocab_size, embedding_dim=300):
    """
    Builds and compiles a Convolutional Neural Network (CNN) model for text classification.

    Args:
        max_length (int): Maximum length of input sequences.
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the embedding layer. Default is 300.

    Returns:
        Sequential: Compiled Keras Sequential model ready for training.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        SpatialDropout1D(0.2),
        Conv1D(128, 5, activation='relu'),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Trains the model with the provided training and validation data.

    Args:
        model (Sequential): The Keras model to be trained.
        X_train (ndarray): Padded training sequences.
        y_train (ndarray): Labels for the training data.
        X_val (ndarray): Padded validation sequences.
        y_val (ndarray): Labels for the validation data.
        epochs (int): Number of epochs for training. Default is 50.
        batch_size (int): Size of each training batch. Default is 32.

    Returns:
        History: Keras History object containing training history.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    return history
