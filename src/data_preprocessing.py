"""
Data Preprocessing Module for Intrusion Detection System

This module handles all data loading and preprocessing tasks including:
- Loading the dataset
- Handling missing values
- Encoding categorical variables
- Feature scaling
- Separating features and labels
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Class for handling data preprocessing operations for the IDS system.
    """
    
    def __init__(self, dataset_path):
        """
        Initialize the DataPreprocessor.
        
        Args:
            dataset_path (str): Path to the dataset CSV file
        """
        self.dataset_path = dataset_path
        self.data = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.X = None
        self.y = None
        
    def load_data(self):
        """
        Load the dataset from the specified path.
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            self.data = pd.read_csv(self.dataset_path)
            logger.info(f"Dataset loaded successfully!")
            logger.info(f"Dataset shape: {self.data.shape}")
            logger.info(f"Column names: {list(self.data.columns)}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def inspect_data(self):
        """
        Perform basic inspection of the dataset.
        
        Returns:
            dict: Dictionary containing inspection results
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        inspection_results = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'missing_percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict(),
            'label_distribution': self.data['label'].value_counts().to_dict() if 'label' in self.data.columns else None
        }
        
        logger.info("\n" + "="*50)
        logger.info("DATA INSPECTION RESULTS")
        logger.info("="*50)
        logger.info(f"Shape: {inspection_results['shape']}")
        logger.info(f"\nMissing Values:\n{self.data.isnull().sum()}")
        logger.info(f"\nLabel Distribution:\n{self.data['label'].value_counts() if 'label' in self.data.columns else 'No label column'}")
        
        return inspection_results
    
    def handle_missing_values(self):
        """
        Handle missing values in the dataset.
        
        Returns:
            pd.DataFrame: Dataset with missing values handled
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Check for missing values
        missing_count = self.data.isnull().sum().sum()
        logger.info(f"Total missing values found: {missing_count}")
        
        if missing_count > 0:
            # For numerical columns, fill with median
            numerical_cols = self.data.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if self.data[col].isnull().sum() > 0:
                    median_value = self.data[col].median()
                    self.data[col].fillna(median_value, inplace=True)
                    logger.info(f"Filled missing values in {col} with median: {median_value}")
            
            # For categorical columns, fill with mode
            categorical_cols = self.data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if self.data[col].isnull().sum() > 0:
                    mode_value = self.data[col].mode()[0]
                    self.data[col].fillna(mode_value, inplace=True)
                    logger.info(f"Filled missing values in {col} with mode: {mode_value}")
        
        logger.info("Missing values handled successfully!")
        return self.data
    
    def encode_categorical(self):
        """
        Encode categorical columns using LabelEncoder.
        
        Returns:
            pd.DataFrame: Dataset with encoded categorical columns
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'label']
        
        logger.info(f"Categorical columns to encode: {list(categorical_cols)}")
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])
            self.label_encoders[col] = le
            logger.info(f"Encoded column: {col}")
        
        logger.info("Categorical encoding completed!")
        return self.data
    
    def prepare_binary_labels(self):
        """
        Convert multi-class labels to binary (normal vs attack).
        
        Returns:
            pd.Series: Binary labels
        """
        if self.data is None or 'label' not in self.data.columns:
            raise ValueError("Data not loaded or label column not found.")
        
        # Convert to binary: normal = 0, attack = 1
        binary_labels = self.data['label'].apply(lambda x: 0 if x == 'normal' else 1)
        
        logger.info("Binary label distribution:")
        logger.info(f"Normal (0): {(binary_labels == 0).sum()}")
        logger.info(f"Attack (1): {(binary_labels == 1).sum()}")
        
        return binary_labels
    
    def separate_features_labels(self, binary_classification=True):
        """
        Separate features and labels.
        
        Args:
            binary_classification (bool): If True, convert labels to binary
        
        Returns:
            tuple: (X, y) where X is features and y is labels
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Separate features and labels
        feature_cols = [col for col in self.data.columns if col != 'label']
        self.X = self.data[feature_cols]
        
        if binary_classification:
            self.y = self.prepare_binary_labels()
        else:
            self.y = self.data['label']
        
        logger.info(f"Features shape: {self.X.shape}")
        logger.info(f"Labels shape: {self.y.shape}")
        
        return self.X, self.y
    
    def scale_features(self, X_train, X_test=None):
        """
        Scale features using StandardScaler.
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
        
        Returns:
            tuple: Scaled X_train and X_test (if provided)
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def get_feature_names(self):
        """
        Get the names of feature columns.
        
        Returns:
            list: List of feature column names
        """
        if self.X is None:
            raise ValueError("Features not separated. Call separate_features_labels() first.")
        
        return list(self.X.columns)
    
    def preprocess_pipeline(self, binary_classification=True):
        """
        Run the complete preprocessing pipeline.
        
        Args:
            binary_classification (bool): If True, use binary labels
        
        Returns:
            tuple: (X, y) preprocessed features and labels
        """
        logger.info("\n" + "="*50)
        logger.info("STARTING DATA PREPROCESSING PIPELINE")
        logger.info("="*50)
        
        # Load data
        self.load_data()
        
        # Inspect data
        self.inspect_data()
        
        # Handle missing values
        self.handle_missing_values()
        
        # Encode categorical variables
        self.encode_categorical()
        
        # Separate features and labels
        X, y = self.separate_features_labels(binary_classification=binary_classification)
        
        logger.info("\n" + "="*50)
        logger.info("PREPROCESSING COMPLETED SUCCESSFULLY")
        logger.info("="*50)
        
        return X, y


if __name__ == "__main__":
    # Test the preprocessing module
    dataset_path = "../dataset/dataset.csv"
    
    preprocessor = DataPreprocessor(dataset_path)
    X, y = preprocessor.preprocess_pipeline(binary_classification=True)
    
    print(f"\nFinal preprocessed data:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Feature names: {preprocessor.get_feature_names()}")
