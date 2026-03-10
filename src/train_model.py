"""
Model Training Module for Intrusion Detection System

This module handles:
- Training multiple machine learning models (Decision Tree, Random Forest)
- Model evaluation using various metrics
- Saving the best performing model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import sys

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import DataPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Class for training and evaluating machine learning models for IDS.
    """
    
    def __init__(self, X, y, test_size=0.2, random_state=42):
        """
        Initialize the ModelTrainer.
        
        Args:
            X: Features
            y: Labels
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
        """
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def split_data(self):
        """
        Split data into training and testing sets.
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state, stratify=self.y
        )
        
        logger.info("Data split completed!")
        logger.info(f"Training set size: {self.X_train.shape[0]} samples")
        logger.info(f"Testing set size: {self.X_test.shape[0]} samples")
        logger.info(f"Training set class distribution:\n{pd.Series(self.y_train).value_counts()}")
        logger.info(f"Testing set class distribution:\n{pd.Series(self.y_test).value_counts()}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_decision_tree(self):
        """
        Train a Decision Tree classifier.
        
        Returns:
            DecisionTreeClassifier: Trained model
        """
        logger.info("\n" + "="*50)
        logger.info("Training Decision Tree Classifier...")
        logger.info("="*50)
        
        dt_params = {
            'criterion': 'gini',
            'max_depth': 20,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': self.random_state
        }
        
        dt_model = DecisionTreeClassifier(**dt_params)
        dt_model.fit(self.X_train, self.y_train)
        
        logger.info("Decision Tree training completed!")
        
        self.models['Decision Tree'] = dt_model
        return dt_model
    
    def train_random_forest(self):
        """
        Train a Random Forest classifier.
        
        Returns:
            RandomForestClassifier: Trained model
        """
        logger.info("\n" + "="*50)
        logger.info("Training Random Forest Classifier...")
        logger.info("="*50)
        
        rf_params = {
            'n_estimators': 100,
            'criterion': 'gini',
            'max_depth': 20,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        rf_model = RandomForestClassifier(**rf_params)
        rf_model.fit(self.X_train, self.y_train)
        
        logger.info("Random Forest training completed!")
        
        self.models['Random Forest'] = rf_model
        return rf_model
    
    def evaluate_model(self, model_name, model):
        """
        Evaluate a trained model.
        
        Args:
            model_name (str): Name of the model
            model: Trained model object
        
        Returns:
            dict: Evaluation metrics
        """
        logger.info(f"\nEvaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(self.X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Classification report
        class_report = classification_report(self.y_test, y_pred, target_names=['Normal', 'Attack'])
        
        # Store results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'predictions': y_pred
        }
        
        self.results[model_name] = results
        
        # Log results
        logger.info(f"\n{model_name} Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        logger.info(f"\nConfusion Matrix:\n{cm}")
        logger.info(f"\nClassification Report:\n{class_report}")
        
        return results
    
    def plot_confusion_matrix(self, model_name, save_path=None):
        """
        Plot confusion matrix for a model.
        
        Args:
            model_name (str): Name of the model
            save_path (str): Path to save the plot
        """
        if model_name not in self.results:
            logger.error(f"Model {model_name} not found in results.")
            return
        
        cm = self.results[model_name]['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Attack'],
                    yticklabels=['Normal', 'Attack'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def compare_models(self):
        """
        Compare all trained models and select the best one.
        
        Returns:
            tuple: (best_model_name, best_model)
        """
        logger.info("\n" + "="*50)
        logger.info("MODEL COMPARISON")
        logger.info("="*50)
        
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        logger.info(f"\n{comparison_df.to_string(index=False)}")
        
        # Select best model based on F1-Score
        best_idx = comparison_df['F1-Score'].idxmax()
        self.best_model_name = comparison_df.loc[best_idx, 'Model']
        self.best_model = self.models[self.best_model_name]
        
        logger.info(f"\nBest Model: {self.best_model_name}")
        logger.info(f"Best F1-Score: {comparison_df.loc[best_idx, 'F1-Score']:.4f}")
        
        return self.best_model_name, self.best_model
    
    def save_best_model(self, model_path):
        """
        Save the best model to disk.
        
        Args:
            model_path (str): Path to save the model
        """
        if self.best_model is None:
            raise ValueError("No best model selected. Run compare_models() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        joblib.dump(self.best_model, model_path)
        logger.info(f"Best model saved to {model_path}")
        
        # Save model info
        model_info = {
            'model_name': self.best_model_name,
            'accuracy': self.results[self.best_model_name]['accuracy'],
            'precision': self.results[self.best_model_name]['precision'],
            'recall': self.results[self.best_model_name]['recall'],
            'f1_score': self.results[self.best_model_name]['f1_score']
        }
        
        info_path = model_path.replace('.pkl', '_info.pkl')
        joblib.dump(model_info, info_path)
        logger.info(f"Model info saved to {info_path}")
    
    def train_all_models(self):
        """
        Train all models and evaluate them.
        
        Returns:
            dict: Results for all models
        """
        logger.info("\n" + "="*50)
        logger.info("TRAINING ALL MODELS")
        logger.info("="*50)
        
        # Train Decision Tree
        self.train_decision_tree()
        self.evaluate_model('Decision Tree', self.models['Decision Tree'])
        
        # Train Random Forest
        self.train_random_forest()
        self.evaluate_model('Random Forest', self.models['Random Forest'])
        
        return self.results
    
    def get_feature_importance(self, model_name, feature_names):
        """
        Get feature importance from a model.
        
        Args:
            model_name (str): Name of the model
            feature_names (list): List of feature names
        
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found.")
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            logger.info(f"\nTop 10 Important Features - {model_name}:")
            logger.info(feature_importance_df.head(10).to_string(index=False))
            
            return feature_importance_df
        else:
            logger.warning(f"Model {model_name} does not have feature_importances_ attribute.")
            return None


def main():
    """
    Main function to train models and save the best one.
    """
    # Paths
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset', 'dataset.csv')
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'ids_model.pkl')
    
    logger.info("Starting Model Training Pipeline...")
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Model save path: {model_path}")
    
    # Preprocess data
    preprocessor = DataPreprocessor(dataset_path)
    X, y = preprocessor.preprocess_pipeline(binary_classification=True)
    feature_names = preprocessor.get_feature_names()
    
    # Train models
    trainer = ModelTrainer(X, y, test_size=0.2, random_state=42)
    trainer.split_data()
    trainer.train_all_models()
    
    # Plot confusion matrices
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    for model_name in trainer.models.keys():
        plot_path = os.path.join(plots_dir, f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
        trainer.plot_confusion_matrix(model_name, save_path=plot_path)
    
    # Get feature importance
    trainer.get_feature_importance('Decision Tree', feature_names)
    trainer.get_feature_importance('Random Forest', feature_names)
    
    # Compare and save best model
    best_name, best_model = trainer.compare_models()
    trainer.save_best_model(model_path)
    
    logger.info("\n" + "="*50)
    logger.info("MODEL TRAINING COMPLETED SUCCESSFULLY")
    logger.info("="*50)
    logger.info(f"Best Model: {best_name}")
    logger.info(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
