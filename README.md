# Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

A machine learning project that detects fraudulent credit card transactions using Logistic Regression with 93% accuracy.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [Author](#author)

---

## Overview

This project implements a binary classification model to identify fraudulent credit card transactions. The model handles highly imbalanced data using under-sampling techniques and achieves strong predictive performance.

### Key Highlights

- **Accuracy**: 93% on test data
- **Algorithm**: Logistic Regression
- **Data Balancing**: Under-sampling technique
- **Dataset Size**: 284,807 transactions

---

## Dataset

The dataset contains transactions made by credit cards in September 2013 by European cardholders.

| Attribute | Value |
|-----------|-------|
| **Total Transactions** | 284,807 |
| **Fraudulent Cases** | 492 (0.172%) |
| **Legitimate Cases** | 284,315 |
| **Features** | 30 (28 PCA-transformed + Time + Amount) |
| **Target Variable** | Class (0 = Legitimate, 1 = Fraud) |

**Download Dataset**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

> **Note**: The dataset is not included in this repository due to its size (143 MB). Download it from Kaggle and place it in the `data/` folder.

---

## Features

- Data exploration and statistical analysis
- Missing value detection
- Class distribution analysis
- Handling imbalanced datasets with under-sampling
- Model training with Logistic Regression
- Comprehensive model evaluation

---

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/suhanijindal/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   - Visit [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
   - Download `creditcard.csv`
   - Place it in the `data/` folder

---

## Usage

Run the training script from the project root:

```bash
cd src
python train.py
```

### What the script does:

1. Loads and explores the dataset
2. Checks for missing values
3. Analyzes class distribution
4. Creates a balanced dataset using under-sampling
5. Splits data into training and testing sets (80-20)
6. Trains a Logistic Regression model
7. Evaluates model performance

---

## Model Performance

### Results

| Metric | Training Data | Testing Data |
|--------|---------------|--------------|
| **Accuracy** | ~94% | ~93% |

### Methodology

**Data Preprocessing**
- Class imbalance ratio: 492 fraud cases vs 284,315 legitimate cases
- Under-sampling: Selected 492 random legitimate cases to match fraud cases
- Final balanced dataset: 984 transactions (492 fraud + 492 legitimate)

**Model Training**
- Algorithm: Logistic Regression
- Train-test split: 80-20 ratio
- Stratified sampling to maintain class distribution

---

## Project Structure

```
credit-card-fraud-detection/
│
├── src/
│   └── train.py              # Main training script
│
├── data/
│   └── creditcard.csv        # Dataset (download separately)
│
├── models/                   # Directory for saved models
│
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore               # Git ignore rules
```

---

## Technologies

- **Python 3.x** - Programming language
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning library
  - Logistic Regression
  - Train-test split
  - Accuracy metrics

---

## Author

**Suhani Jindal**

GitHub: [@suhanijindal](https://github.com/suhanijindal)

---

## Acknowledgments

- Dataset provided by [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Inspired by real-world fraud detection systems
- Built with scikit-learn and Python

---

**⭐ If you find this project useful, please consider giving it a star!**