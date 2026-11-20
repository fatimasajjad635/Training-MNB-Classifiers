#!/usr/bin/env python3
"""
MSCI 641 Assignment 2: Text Classification with Multinomial Naive Bayes
-----------------------------------------
Author: Fatima Sajjad
Student ID: 20718749
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

def load_data_and_labels(data_file, label_file):
    """Load data and labels simultaneously, discarding any empty or invalid lines."""
    data = []
    labels = []
    with open(data_file, 'r', encoding='utf-8') as df, open(label_file, 'r', encoding='utf-8') as lf:
        for data_line, label_line in zip(df, lf):
            tokens = data_line.strip().split(',')
            if tokens and any(tokens):  # Only keep non-empty lines
                data.append(tokens)
                labels.append(label_line.strip())
    return data, labels

def train_model(train_data, train_labels, use_bigrams=False, use_unigrams=True):
    """
    Train a Multinomial Naive Bayes classifier.

    Args:
        train_data: List of texts to train on
        train_labels: List of corresponding labels
        use_bigrams: Whether to include bigram features
        use_unigrams: Whether to include unigram features

    Returns:
        A trained model that can be used for prediction
    """
    if use_unigrams and use_bigrams:
        ngram_range = (1, 2)
    elif use_bigrams:
        ngram_range = (2, 2)
    else:
        ngram_range = (1, 1)
     
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=ngram_range)),
        ('classifier', MultinomialNB())
    ])
    pipeline.fit(train_data, train_labels)
    return pipeline
"""
    Implementation steps:
    1. Configure the n-gram range based on the feature requirements
       - If using only unigrams, set range to (1,1)
       - If using only bigrams, set range to (2,2)
       - If using both, set range to (1,2)
       
    2. Create a pipeline with two steps:
       - A text vectorizer that converts text to token counts with the configured n-gram range
       - A Multinomial Naive Bayes classifier
       
    3. Train the pipeline on the training data and labels
    
    4. Return the trained pipeline
    """
    # Your implementation here
    
from sklearn.metrics import accuracy_score

def evaluate_model(model, test_data, test_labels):
    """
    Evaluate a model on test data

    Args:
        model: A trained classifier model
        test_data: List of texts to evaluate on
        test_labels: List of true labels for the test data

    Returns:
        float: The accuracy of the model on the test data
    """
    predictions = model.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)
    return accuracy

import os
import pickle
import csv

def main():
    configurations = [
    (True, False, False, "unigrams_with_stopwords"),
    (False, True, False, "bigrams_with_stopwords"),
    (True, True, False, "unigrams_bigrams_with_stopwords"),
    (True, False, True, "unigrams_without_stopwords"),
    (False, True, True, "bigrams_without_stopwords"),
    (True, True, True, "unigrams_bigrams_without_stopwords")
]

    results = []

    for use_unigrams, use_bigrams, stopwords_removed, name in configurations:
        data_suffix = "_ns" if stopwords_removed else ""
        train_file = f"data/train{data_suffix}.csv"
        test_file = f"data/test{data_suffix}.csv"
        label_file = "data/train_labels.csv"
        test_label_file = "data/test_labels.csv"

        train_data, train_labels = load_data_and_labels(train_file, label_file)
        test_data, test_labels = load_data_and_labels(test_file, test_label_file)

        train_data = [" ".join(tokens) for tokens in train_data]
        test_data = [" ".join(tokens) for tokens in test_data]

        model = train_model(train_data, train_labels,
                        use_bigrams=use_bigrams, use_unigrams=use_unigrams)

        accuracy = evaluate_model(model, test_data, test_labels)
        print(f"Finished training {name} with accuracy: {accuracy}")

        model_filename = f"models/{name}.pkl"
        with open(model_filename, "wb") as f:
            pickle.dump(model, f)

        results.append({
    "Stopwords removed": "yes" if stopwords_removed else "no",
    "text features": name,
    "Accuracy (test set)": accuracy
    })

    with open("results.csv", "w", newline='') as csvfile:
        fieldnames = ["Stopwords removed", "text features", "Accuracy (test set)"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

if __name__ == "__main__":
    main()
