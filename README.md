# AI-Based Fraud Detection in Credit Card Transactions

## Overview

This project implements an AI-based fraud detection system for credit card transactions using machine learning algorithms. The model is trained to identify fraudulent transactions with high accuracy using various classification techniques.

## Features

- **Data Preprocessing**: Handles missing values, outliers, and feature scaling
- **Exploratory Data Analysis**: Comprehensive EDA with visualizations
- **Multiple ML Models**: Random Forest, Logistic Regression, and Ensemble methods
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Confusion Matrix & Visualization**: Detailed model evaluation
- **Imbalanced Data Handling**: SMOTE technique for handling class imbalance

## Dataset

The project uses the Kaggle Credit Card Fraud Detection dataset:
- **Total Transactions**: 284,807
- **Fraudulent Transactions**: 492 (0.172%)
- **Features**: 31 (V1-V28, Time, Amount, Class)

## Installation & Setup

### Requirements
```bash
python >= 3.8
pandas
numpy
scikit-learn
matplotlib
seaborn
imbalanced-learn
```

### Installation
```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
```

## Usage

### Option 1: Google Colab (Recommended)
1. Open the `Fraud_Detection_Notebook.ipynb` in Google Colab
2. Run all cells sequentially
3. The dataset will be automatically downloaded from Kaggle

### Option 2: Local Environment
1. Clone the repository
2. Install dependencies
3. Run the notebook or Python script

## Project Structure

```
.
├── README.md
├── fraud_detection_notebook.ipynb
├── fraud_detection.py
└── requirements.txt
```

## Model Performance

### Results Summary
- **Logistic Regression**: ~95% Accuracy
- **Random Forest**: ~99.9% Accuracy
- **ROC-AUC Score**: 0.98+

## Key Insights

1. **Class Imbalance**: Fraudulent transactions are extremely rare (0.17%)
2. **Feature Importance**: V4, V12, V14 are top predictive features
3. **SMOTE Effectiveness**: Significantly improves recall for fraud cases
4. **Random Forest**: Best performer with minimal false positives

## Future Improvements

- Implement Deep Learning models (Neural Networks, LSTM)
- Real-time prediction API
- Model deployment on cloud platforms
- Continuous model retraining pipeline

## Author

Kishan Halageri

## License

MIT License

## References

- Kaggle Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Documentation on SMOTE: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
