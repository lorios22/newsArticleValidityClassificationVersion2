import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

def tokenize_text(X_train, X_val, max_words=10000, max_len=100):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_len)
    X_val_padded = pad_sequences(X_val_seq, maxlen=max_len)
    return X_train_padded, X_val_padded, tokenizer

def build_model(max_length=100):
    model = Sequential([
        Embedding(input_dim=10000, output_dim=128, input_length=max_length),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3)),
        Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))
    ])
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

def train_model(model, X_train_padded, y_train, X_val_padded, y_val):
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    history = model.fit(X_train_padded, y_train, epochs=20, batch_size=32, validation_data=(X_val_padded, y_val), callbacks=[early_stopping])
    return history, model