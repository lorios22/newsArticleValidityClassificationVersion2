# News Article Validity Classification Project

## Introduction

This project aims to build a machine learning model that classifies news articles as either 'valid' or 'invalid'. The model is designed using natural language processing (NLP) techniques, specifically a Long Short-Term Memory (LSTM) network, and incorporates explainable artificial intelligence (XAI) methods for model interpretability.

In this project, we will walk through the process of data preprocessing, model training, evaluation, and explanation of the model's predictions. This README file provides a comprehensive overview of the project, including its objectives, structure, and usage instructions.

## Project objectives
Data preprocessing: To prepare the raw data for training the model by cleaning and structuring the data.
Model development: To create a predictive model for classifying news articles based on their validity.
Model evaluation: To assess the performance of the model using various metrics and visualize the training process.
Explainability: To generate and interpret explanations for the model's predictions using SHAP (SHapley Additive explanations).

## Project structure
Here is a detailed overview of the project's directory structure and the purpose of each file:

preprocessing/preprocess.py:
Functions: load_and_process_data, clean_text
Description: Contains functions for loading data from a JSONL file, cleaning text data, and splitting the data into training and validation sets.

models/train.py:
Functions: tokenize_text, build_model, train_model
Description: Includes functions for text tokenization, building the LSTM-based model, and training the model with the data.

models/evaluate.py:
Functions: evaluate_model
Description: Contains functions for evaluating the model's performance using metrics such as accuracy, precision, recall, and F1-score.

utils/model_utils.py:
Functions: load_model_and_tokenizer, preprocess_text, model_predict
Description: Provides utilities for loading the trained model and tokenizer, preprocessing text for predictions, and making predictions.

utils/explain_utils.py:
Functions: generate_explanation, word_importance
Description: Includes functions for generating SHAP explanations for model predictions and calculating word importance.

exloration.ipynb:
Description: A Jupyter Notebook that is used to explore the dataset.

News_Article_Validity_Classification.ipynb:
Description: A Jupyter Notebook that integrates the above scripts to perform data processing, model training, evaluation, and explanation.

## Installation
To get started with the project, follow these steps:

1. Clone the Repository:
```bash
git clone https://github.com/yourusername/news-article-validity-classification.git
cd news-article-validity-classification
```

2. Install Dependencies:

Ensure you have Python 3.7 or later installed. Create a virtual environment (recommended) and install the required libraries:

```bash
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
```

## Usage:

### Running the Jupyter Notebook

1. Start Jupyter Notebook:

Navigate the directory and start Jupyter Notebook:

2. Open the News_Article_Validity_Classification.ipynb notebook and execute the cells in order to:

- Load and process data: Load data from the data/fine_data.jsonl file, clean it, and split it into training and validation sets.
- Preprocess text data: Tokenize the text data and prepare it for model training.
- Build and train the model: Define and train the LSTM-based model for classifying news articles.
- Evaluate the model: Plot training history for accuracy and loss, and evaluate the model’s performance using various metrics.
- Generate Explanations: Use SHAP to provide explanations for the model’s predictions and identify the most influential words. 
