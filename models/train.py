import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional, GlobalMaxPooling1D, Conv1D
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

def tokenize_text(X_train, X_val, max_words=50000, max_len=400):
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_val_padded = pad_sequences(X_val_seq, maxlen=max_len, padding='post', truncating='post')
    return X_train_padded, X_val_padded, tokenizer

def build_model(max_length=400, vocab_size=50000):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=300, input_length=max_length),
        SpatialDropout1D(0.3),
        Conv1D(128, 5, activation='relu'),
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
        Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
        GlobalMaxPooling1D(),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def train_model(model, X_train_padded, y_train, X_val_padded, y_val, epochs=50):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
    
    history = model.fit(
        X_train_padded, y_train, 
        epochs=epochs, 
        batch_size=32, 
        validation_data=(X_val_padded, y_val), 
        callbacks=[early_stopping, reduce_lr]
    )
    return history, model