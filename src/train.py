"""
Credit Card Fraud Detection - Training Script
Author: Suhani
Description: Train a Logistic Regression model to detect fraudulent credit card transactions
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Load dataset
print("Loading dataset...")
df = pd.read_csv("../data/creditcard.csv")

print("Dataset loaded!")
print(df.head())
print("Shape:", df.shape)

# Check for missing values
print("\nChecking for missing values...")
missing_values = df.isnull().sum()
print(missing_values)

# Distribution of legit and fraud cases
print("\nDistribution of legit and fraud cases:")
print(df['Class'].value_counts())
print("Highly imbalanced dataset detected.")

# Separate the data
legit = df[df.Class == 0]
fraud = df[df.Class == 1]

print("\nLegit cases shape:", legit.shape)
print("Fraud cases shape:", fraud.shape)

# Statistical measures of the data
print("\nStatistical measures of legit cases:")
print(legit.describe())
print("\nStatistical measures of fraud cases:")
print(fraud.describe())

# Compare the mean values for both cases
print("\nComparing mean values of features for legit and fraud cases:")
print(df.groupby('Class').mean())

# Under-sampling: build a balanced dataset
# Number of fraud cases = 492
print("\nCreating balanced dataset using under-sampling...")
legit_sample = legit.sample(n=492, random_state=2)

# Concatenate dataframes
new_df = pd.concat([legit_sample, fraud], axis=0)

print("New dataset class distribution:")
print(new_df['Class'].value_counts())
print("\nMean values in balanced dataset:")
print(new_df.groupby('Class').mean())

# Split data into features and target
X = new_df.drop(columns='Class')
Y = new_df['Class']

# Split into training and testing sets
print("\nSplitting data into training and testing sets...")
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Model training
print("\nTraining Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
print("Model training completed!")

# Model evaluation
print("\nEvaluating model's accuracy score...")

# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print("Accuracy on training data:", training_data_accuracy * 100, "%")

# Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print("Accuracy on test data:", test_data_accuracy * 100, "%")

print("\nTraining complete!")







