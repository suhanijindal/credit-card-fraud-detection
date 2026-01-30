# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using Logistic Regression.

## Overview

This project uses a dataset of credit card transactions to train a binary classification model that identifies fraudulent transactions. The dataset is highly imbalanced, with fraudulent transactions making up only a small percentage of the total data.

## Features

- Data exploration and analysis
- Handling imbalanced dataset using under-sampling
- Logistic Regression model training
- Model evaluation with accuracy metrics

## Project Structure

```
credit-card-fraud-detection/
├── src/
│   └── train.py          # Main training script
├── data/
│   └── creditcard.csv    # Dataset
├── models/               # Saved models directory
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
   - Download from [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
   - Place `creditcard.csv` in the project root directory

## Usage

Run the training script:

```bash
cd src
python train.py
```

The script will:
1. Load and explore the dataset
2. Analyze class distribution
3. Create a balanced dataset using under-sampling
4. Train a Logistic Regression model
5. Evaluate model performance

## Dataset

The dataset contains transactions made by credit cards in September 2013 by European cardholders.

- Total transactions: 284,807
- Fraudulent transactions: 492 (0.172%)
- Features: 30 (28 PCA-transformed features + Time + Amount)
- Target: Class (0 = legitimate, 1 = fraudulent)

## Model Performance

The Logistic Regression model achieves:
- Training accuracy: ~94%
- Testing accuracy: ~93%

## Methodology

### 1. Data Preprocessing
- Check for missing values
- Analyze class distribution
- Statistical analysis of features

### 2. Handling Imbalanced Data
- Under-sampling technique used
- Sample 492 legitimate transactions to match fraud cases
- Creates balanced dataset of 984 total transactions

### 3. Model Training
- Algorithm: Logistic Regression
- Train-test split: 80-20
- Stratified sampling to maintain class distribution

### 4. Evaluation
- Accuracy score on training data
- Accuracy score on testing data

## Technologies Used

- Python 3.x
- NumPy
- Pandas
- Scikit-learn

## Future Improvements

- Try other algorithms (Random Forest, XGBoost, Neural Networks)
- Implement over-sampling techniques (SMOTE)
- Add more evaluation metrics (Precision, Recall, F1-Score, ROC-AUC)
- Cross-validation for better model assessment
- Feature importance analysis

## License

This project is open source and available under the MIT License.

## Author

Suhani

## Acknowledgments

- Dataset provided by Kaggle
- Inspired by real-world fraud detection systems