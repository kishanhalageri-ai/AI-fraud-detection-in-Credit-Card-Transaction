"""
Google Colab Setup Script for AI Fraud Detection

This script is optimized to run in Google Colab environment.
It handles dataset download, model training, and evaluation.

Usage:
1. Open Google Colab: https://colab.research.google.com
2. Upload this file or copy the code cells below
3. Run the cells sequentially

Author: Kishan Halageri
"""

# ============================================================================
# CELL 1: Install Required Libraries
# ============================================================================

!pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn kaggle

# ============================================================================
# CELL 2: Import Libraries
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("All libraries imported successfully!")

# ============================================================================
# CELL 3: Download Dataset from Kaggle
# ============================================================================

# Option 1: Manual Upload (Recommended for first-time users)
# Uncomment and run this if you want to upload the CSV manually

from google.colab import files
print("Upload your creditcard.csv file:")
uploaded = files.upload()
data_path = list(uploaded.keys())[0]
print(f"File uploaded: {data_path}")

# Option 2: Download from Kaggle API (Requires API key)
# Uncomment this section if you have Kaggle API credentials
# !mkdir -p ~/.kaggle
# files.upload()  # Upload your kaggle.json
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d mlg-ulb/creditcardfraud
# !unzip creditcardfraud.zip
# data_path = 'creditcard.csv'

# ============================================================================
# CELL 4: Load and Explore Data
# ============================================================================

data = pd.read_csv(data_path)

print("Dataset Shape:", data.shape)
print("\nFirst few rows:")
print(data.head())
print("\nData Info:")
print(data.info())
print("\nClass Distribution:")
print(data['Class'].value_counts())
print("\nClass Percentage:")
print(data['Class'].value_counts(normalize=True) * 100)
print("\nBasic Statistics:")
print(data.describe())

# ============================================================================
# CELL 5: Data Preprocessing
# ============================================================================

# Separate features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Split data (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled successfully!")

# ============================================================================
# CELL 6: Handle Class Imbalance with SMOTE
# ============================================================================

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"After SMOTE:")
print(f"Training set shape: {X_train_resampled.shape}")
print(f"Class distribution:")
print(pd.Series(y_train_resampled).value_counts())

# ============================================================================
# CELL 7: Train Models
# ============================================================================

print("Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_resampled, y_train_resampled)
print("Logistic Regression trained!")

print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_resampled, y_train_resampled)
print("Random Forest trained!")

# ============================================================================
# CELL 8: Model Evaluation
# ============================================================================

print("="*60)
print("LOGISTIC REGRESSION RESULTS")
print("="*60)

lr_pred = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)
lr_roc_auc = roc_auc_score(y_test, lr_model.predict_proba(X_test_scaled)[:, 1])

print(f"Accuracy:  {lr_accuracy:.4f}")
print(f"Precision: {lr_precision:.4f}")
print(f"Recall:    {lr_recall:.4f}")
print(f"F1-Score:  {lr_f1:.4f}")
print(f"ROC-AUC:   {lr_roc_auc:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, lr_pred))

print("\n" + "="*60)
print("RANDOM FOREST RESULTS")
print("="*60)

rf_pred = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test_scaled)[:, 1])

print(f"Accuracy:  {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall:    {rf_recall:.4f}")
print(f"F1-Score:  {rf_f1:.4f}")
print(f"ROC-AUC:   {rf_roc_auc:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, rf_pred))

# ============================================================================
# CELL 9: Confusion Matrix Visualization
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Logistic Regression Confusion Matrix
lr_cm = confusion_matrix(y_test, lr_pred)
sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Logistic Regression - Confusion Matrix')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

# Random Forest Confusion Matrix
rf_cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Random Forest - Confusion Matrix')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=100, bbox_inches='tight')
plt.show()

print("Confusion matrices saved!")

# ============================================================================
# CELL 10: Performance Comparison Visualization
# ============================================================================

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
lr_scores = [lr_accuracy, lr_precision, lr_recall, lr_f1, lr_roc_auc]
rf_scores = [rf_accuracy, rf_precision, rf_recall, rf_f1, rf_roc_auc]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metrics))
width = 0.35

ax.bar(x - width/2, lr_scores, width, label='Logistic Regression', color='skyblue')
ax.bar(x + width/2, rf_scores, width, label='Random Forest', color='lightcoral')

ax.set_xlabel('Metrics')
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=100, bbox_inches='tight')
plt.show()

print("Model comparison plot saved!")

# ============================================================================
# CELL 11: Feature Importance (Random Forest)
# ============================================================================

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'][:10], feature_importance['Importance'][:10])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=100, bbox_inches='tight')
plt.show()

print("Feature importance plot saved!")

# ============================================================================
# CELL 12: Final Summary
# ============================================================================

print("\n" + "="*60)
print("PROJECT SUMMARY")
print("="*60)

if rf_roc_auc > lr_roc_auc:
    best_model = "Random Forest"
    best_score = rf_roc_auc
else:
    best_model = "Logistic Regression"
    best_score = lr_roc_auc

print(f"\nBest Model: {best_model}")
print(f"Best ROC-AUC Score: {best_score:.4f}")
print(f"\nDataset: {data.shape[0]} transactions")
print(f"Fraudulent: {(y == 1).sum()} transactions ({(y == 1).sum()/len(y)*100:.2f}%)")
print(f"Legitimate: {(y == 0).sum()} transactions ({(y == 0).sum()/len(y)*100:.2f}%)")
print("\nFraud detection model training completed successfully!")
print("Check the saved plots for visualizations.")
