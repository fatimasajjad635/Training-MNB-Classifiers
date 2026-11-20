
This guide explains how to complete, which involves training Multinomial Naïve Bayes (MNB) classifiers on the Amazon reviews dataset and submitting your trained models.

## Overview

In this assignment, you will:

1. Train MNB classifiers with different configurations
2. Evaluate the models on your test set
3. Upload your trained models to the university server
4. Submit your code and results to GitHub Classroom

## Prerequisites

- Completed Assignment 1 (data preparation)
- Python 3.10+ with scikit-learn, numpy, and pandas installed
- An API key for the model server (provided by the instructor). A unique API key was generated for each student and is available on Learn > Grades > API Key.

### Opening in Codespaces

1. Navigate to your assignment repository on GitHub
2. Click the green "Code" button
3. Select the "Codespaces" tab
4. Click "Create codespace on main"

## Step 1: Getting Your Data into GitHub Codespaces

To bring your preprocessed data from Assignment 1 into GitHub Codespaces for Assignment 2, follow these steps:

1. Create a directory named `data` in your Codespace:
   ```bash
   mkdir -p data
   ```
2. Click the Explorer icon in the left sidebar
3. Right-click on the `data` directory and select "Upload..."
4. Upload all your preprocessed CSV files from Assignment 1 (train.csv, train_ns.csv, test.csv, test_ns.csv, etc.)

Note, that the `data` directory contains large files, it has been added to .gitignore, and therefore will not be uploaded to GitHub.

### Set up a Python virtual environment and install the required dependencies, if you use a local machine. Note that Codespaces installs all the dependencies in requirements.txt during the initialization, so you can skip this step.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
## Step 2: Set Up the Model Upload Client

```bash
# Download the model upload client
wget https://mse-msci-641.uwaterloo.ca/static/model_upload_client.py

# Initialize the client with your API key
python model_upload_client.py --init
# (Enter the API key provided by the instructor)
```

## Step 3: Implement the MNB Classifier and Train Your Models

You need to train six different MNB classifier configurations:

1. Without stopwords removed (using your `train.csv` and `test.csv` files):
   - Using unigrams
   - Using bigrams
   - Using unigrams + bigrams

2. With stopwords removed (using your `train_ns.csv` and `test_ns.csv` files):
   - Using unigrams
   - Using bigrams
   - Using unigrams + bigrams

## Implementation Guidelines for `main.py`

The starter code provides function signatures and docstrings for the key components of this assignment. Your task is to complete the implementation of these functions and add the main execution flow.

### 1. DATA LOADING FUNCTIONS (ALREADY PROVIDED)
```python
def load_data(file_path):
    """Load tokenized data from a CSV file"""
    # Implementation already provided
```

```python
def load_labels(file_path):
    """Load labels from a CSV file"""
    # Implementation already provided
```
- Review these functions to understand how they work
- Consider if they need enhancement to handle edge cases

### 2. MODEL TRAINING FUNCTION (NEEDS IMPLEMENTATION)
```python
def train_model(train_data, train_labels, use_bigrams=False, use_unigrams=True):
    """Train a Multinomial Naive Bayes classifier"""
    # Your implementation needed here
```
- Follow the implementation steps in the function's docstring
- Configure n-gram features based on the parameters
- Create a pipeline with CountVectorizer and MultinomialNB
- Train and return the model

### 3. MODEL EVALUATION FUNCTION (NEEDS IMPLEMENTATION)
```python
def evaluate_model(model, test_data, test_labels):
    """Evaluate a model on test data"""
    # Your implementation needed here
```
- Follow the implementation steps in the function's docstring
- Generate predictions and calculate accuracy

### 4. MAIN FUNCTION (NEEDS IMPLEMENTATION)
```python
def main():
    # Your implementation needed here
```
Implement the main execution flow:
- Define the six required configurations (unigrams/bigrams with/without stopwords)
- For each configuration:
  * Load appropriate train and test data using the provided functions
  * Preprocess data (convert tokens to strings suitable for CountVectorizer)
  * Train the model using your implemented train_model function
  * Evaluate on test data using your implemented evaluate_model function
  * Save the trained model to disk
  * Record results including stopwords setting, feature type, and accuracy
- Save all results to results.csv in the required format

## Step 4: Upload Your Trained Models

After training your models, upload them to the university server:

```bash
# Upload all models in the 'models' directory
python model_upload_client.py --models_dir models

# This will save a model_ids.json file with identifiers for your uploaded models
```

Check if there are any errors with uploading any of your model files. If so, upload the models individually as follows:

```bash
# Upload a single model, for example:
python model_upload_client.py --model models/unigrams_with_stopwords.pkl --config unigrams_with_stopwords
```

If you upload the same configuration multiple times, only the latest model with the given configuration will be saved.

If you need to delete a previously uploaded model, do the following steps:

```bash
# See the list of all uploaded models
python model_upload_client.py --list

# Copy the name of the model you want to delete
# Delete the model, for example:
python model_upload_client.py --delete username_unigrams_with_stopwords_34690db9
```

If you need to delete all your previously uploaded models on the university server, use this command. Note that the deleted files cannot be restored.

```bash
# Delete all your uploaded models
python model_upload_client.py --delete-all
```

## Create model_ids.json file

Run the following command to generate the complete `model_ids.json` file:

```bash
# Generate model_ids.json file containing the names of all your uploaded models
python model_upload_client.py --generate-json
```

## Step 5: Submit Your Assignment

1. Make sure your repository contains:
   - `main.py` - Your implementation
   - `results.csv` - Your classification results
   - `model_ids.json` - References to your uploaded models

2. Commit and push your changes to GitHub:
   ```bash
   git add main.py results.csv model_ids.json
   git commit -m "Complete Assignment 2"
   git push
   ```

3. The autograder will:
   - Download your models from the server using the IDs in `model_ids.json` 
   - Test your code and models for correctness
   - Verify your results format

## Expected Format for Results

Your `results.csv` file should have exactly three columns:
- `Stopwords removed`: Values should be either "yes" or "no"
- `text features`: Values should be one of "unigrams", "bigrams", or "unigrams+bigrams"
- `Accuracy (test set)`: Values should be between 0 and 1, representing the accuracy on the test set

Example:
```
Stopwords removed,text features,Accuracy (test set)
no,unigrams,0.825
no,bigrams,0.784
no,unigrams+bigrams,0.836
yes,unigrams,0.816
yes,bigrams,0.772
yes,unigrams+bigrams,0.827
```

## Grading Criteria

Your assignment will be graded on:
- Correct implementation of the MNB classifiers (30%)
- Valid model uploads (20%)
- Proper results format and reasonable accuracy values (20%)
- Code quality and organization (20%)
- Correct handling of the different configurations (10%)

## Troubleshooting

### Model Upload Issues

If you have trouble uploading your models:

```bash
# Check if your API key is properly set up
python model_upload_client.py --init

# List your uploaded models
python model_upload_client.py --list

# Try uploading a single model
python model_upload_client.py --model models/unigrams_with_stopwords.pkl --config unigrams_with_stopwords
```

### Model Size Issues

If your models are too large (>500MB), check your implementation:
- MNB models should be relatively small
- Very large models usually indicate an issue with the feature extraction

### GitHub Autograding Issues

If autograding fails:
- Check that your `model_ids.json` file is correctly formatted
- Ensure your `results.csv` follows the required format
- Verify that your models were successfully uploaded

### Common Problems

1. **Wrong file format**: Make sure your data files are in the correct format (comma-separated tokens)

2. **Missing files**: Ensure all required data files exist in the `data` directory

3. **API key issues**: If you encounter authorization issues, re-initialize your API key

4. **Memory errors**: If you run into memory issues when training, consider using a subset of the data for debugging

### Expected Training Times

- **Unigram models**: ~1-2 minutes
- **Bigram models**: ~1-3 minutes
- **Unigram+Bigram models**: ~2-4 minutes

These times are estimates and may vary based on Codespaces resource availability.

### Strategies for Efficient Development

1. **Incremental Development with Data Subsets**: 
   - First test your code on a small subset of data (e.g., 5-10% of the training data)
   - Only run on the full dataset once you're confident in your code
   - For final submission, ensure you train on the full dataset

3. **Local Development Alternative**:
   - If you have access to a more powerful machine, you can develop locally
   - Push your code to GitHub when ready for final testing and submission
   - Just ensure your final version works in Codespaces for the autograder
