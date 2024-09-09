from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_val_padded, y_val):
    """
    Evaluates the model's performance on a validation dataset.

    This function uses the model to predict labels for the validation dataset
    and calculates various evaluation metrics: accuracy, precision, recall, and F1-score.

    Args:
        model (tf.keras.Model): Trained model to make predictions.
        X_val_padded (numpy array): Validation data, padded sequences of integers.
        y_val (numpy array): True labels for the validation dataset.

    Returns:
        tuple: A tuple containing:
            - accuracy (float): Accuracy of the model on the validation set.
            - precision (float): Precision of the model on the validation set.
            - recall (float): Recall of the model on the validation set.
            - f1 (float): F1-score of the model on the validation set.

    Example:
        >>> accuracy, precision, recall, f1 = evaluate_model(model, X_val_padded, y_val)
        >>> print(f"Accuracy: {accuracy}")
        >>> print(f"Precision: {precision}")
        >>> print(f"Recall: {recall}")
        >>> print(f"F1 Score: {f1}")
    """
    y_pred = (model.predict(X_val_padded) > 0.5).astype("int32")
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    return accuracy, precision, recall, f1