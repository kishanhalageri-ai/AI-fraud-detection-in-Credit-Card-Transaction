"""AI-Based Fraud Detection in Credit Card Transactions

This script implements a machine learning model to detect fraudulent credit card transactions.
It uses the Kaggle Credit Card Fraud Detection dataset and provides comprehensive
data analysis, model training, and evaluation.

Author: Kishan Halageri
Date: 2025
"""

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
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class FraudDetectionModel:
    """Fraud Detection Model using Machine Learning"""
    
    def __init__(self, data_path=None):
        """Initialize the fraud detection model"""
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
    def load_data(self, data_path):
        """Load the credit card fraud dataset"""
        print(f"Loading data from {data_path}...")
        self.data = pd.read_csv(data_path)
        print(f"Dataset shape: {self.data.shape}")
        print(f"\nFirst few rows:")
        print(self.data.head())
        print(f"\nData Info:")
        print(self.data.info())
        return self.data
    
    def exploratory_data_analysis(self):
        """Perform comprehensive EDA"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Check for missing values
        print(f"\nMissing values:\n{self.data.isnull().sum().sum()} total missing values")
        
        # Class distribution
        print(f"\nClass Distribution:")
        print(self.data['Class'].value_counts())
        print(f"\nPercentage:")
        print(self.data['Class'].value_counts(normalize=True) * 100)
        
        # Basic statistics
        print(f"\nBasic Statistics:")
        print(self.data.describe())
        
        return True
    
    def preprocess_data(self):
        """Preprocess the data for modeling"""
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Separate features and target
        X = self.data.drop('Class', axis=1)
        y = self.data['Class']
        
        # Train-test split (80-20)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set size: {self.X_train.shape}")
        print(f"Testing set size: {self.X_test.shape}")
        
        # Feature scaling
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
        print("Features scaled using StandardScaler")
        
        return True
    
    def handle_imbalance(self):
        """Handle class imbalance using SMOTE"""
        print("\n" + "="*50)
        print("HANDLING CLASS IMBALANCE WITH SMOTE")
        print("="*50)
        
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        
        print(f"After SMOTE:")
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Class distribution:\n{pd.Series(self.y_train).value_counts()}")
        
        return True
    
    def train_models(self):
        """Train multiple models"""
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        # Logistic Regression
        print("\nTraining Logistic Regression...")
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = lr_model
        print("Logistic Regression training completed")
        
        # Random Forest
        print("\nTraining Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf_model
        print("Random Forest training completed")
        
        return self.models
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        for model_name, model in self.models.items():
            print(f"\n{model_name}:")
            print("-" * 40)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            self.results[model_name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"ROC-AUC:   {roc_auc:.4f}")
            
            print(f"\nClassification Report:")
            print(classification_report(self.y_test, y_pred))
        
        return self.results
    
    def plot_results(self):
        """Visualize model results"""
        print("\nGenerating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy Comparison
        models = list(self.results.keys())
        accuracies = [self.results[m]['Accuracy'] for m in models]
        axes[0, 0].bar(models, accuracies, color='skyblue')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.01, f'{v:.4f}', ha='center')
        
        # Plot 2: Precision vs Recall
        precisions = [self.results[m]['Precision'] for m in models]
        recalls = [self.results[m]['Recall'] for m in models]
        x = np.arange(len(models))
        width = 0.35
        axes[0, 1].bar(x - width/2, precisions, width, label='Precision', color='green')
        axes[0, 1].bar(x + width/2, recalls, width, label='Recall', color='orange')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Precision vs Recall')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(models)
        axes[0, 1].legend()
        
        # Plot 3: F1-Score Comparison
        f1_scores = [self.results[m]['F1-Score'] for m in models]
        axes[1, 0].bar(models, f1_scores, color='coral')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].set_title('F1-Score Comparison')
        axes[1, 0].set_ylim(0, 1)
        for i, v in enumerate(f1_scores):
            axes[1, 0].text(i, v + 0.01, f'{v:.4f}', ha='center')
        
        # Plot 4: ROC-AUC Comparison
        roc_aucs = [self.results[m]['ROC-AUC'] for m in models]
        axes[1, 1].bar(models, roc_aucs, color='purple')
        axes[1, 1].set_ylabel('ROC-AUC')
        axes[1, 1].set_title('ROC-AUC Comparison')
        axes[1, 1].set_ylim(0, 1)
        for i, v in enumerate(roc_aucs):
            axes[1, 1].text(i, v + 0.01, f'{v:.4f}', ha='center')
        
        plt.tight_layout()
        plt.savefig('model_performance.png', dpi=100, bbox_inches='tight')
        print("Model performance plot saved as 'model_performance.png'")
        plt.show()
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices"""
        fig, axes = plt.subplots(1, len(self.results), figsize=(14, 4))
        fig.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            cm = confusion_matrix(self.y_test, self.results[model_name]['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(model_name)
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=100, bbox_inches='tight')
        print("Confusion matrices saved as 'confusion_matrices.png'")
        plt.show()
    
    def print_summary(self):
        """Print model summary"""
        print("\n" + "="*50)
        print("FINAL SUMMARY")
        print("="*50)
        
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.drop(['y_pred', 'y_pred_proba'], axis=1)
        print("\n", results_df)
        
        best_model = max(self.results, key=lambda x: self.results[x]['ROC-AUC'])
        print(f"\nBest Model (by ROC-AUC): {best_model}")
        print(f"Best ROC-AUC Score: {self.results[best_model]['ROC-AUC']:.4f}")


def main():
    """Main execution function"""
    print("\nAI-Based Fraud Detection in Credit Card Transactions")
    print("="*50)
    
    # Initialize model
    model = FraudDetectionModel()
    
    # Load data (update path as needed)
    # data_path = 'creditcard.csv'
    # model.load_data(data_path)
    
    # For Google Colab users:
    # from google.colab import files
    # files.upload()  # Upload the CSV file
    # model.load_data('creditcard.csv')
    
    # Perform EDA
    # model.exploratory_data_analysis()
    
    # Preprocess data
    # model.preprocess_data()
    
    # Handle imbalance
    # model.handle_imbalance()
    
    # Train models
    # model.train_models()
    
    # Evaluate models
    # model.evaluate_models()
    
    # Plot results
    # model.plot_results()
    # model.plot_confusion_matrices()
    
    # Print summary
    # model.print_summary()
    
    print("\nFraud Detection Pipeline Ready!")
    print("To use in Google Colab, uncomment the main() function calls above.")


if __name__ == "__main__":
    main()
