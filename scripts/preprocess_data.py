"""
Data Preprocessing Pipeline for Toxic Comment Detection
Cleans text data, handles class imbalance, and prepares data for training
"""

import pandas as pd
import numpy as np
import re
import string
from typing import List, Tuple
import logging
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    logger.info("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class DataPreprocessor:
    """Comprehensive data preprocessing for toxic comment detection"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text data"""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers (optional - keep for some contexts)
        # text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatization
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def load_and_clean_data(self, csv_path: str, sample_size: int = None) -> pd.DataFrame:
        """Load and clean the dataset"""
        logger.info(f"Loading dataset from {csv_path}")
        df = pd.read_csv(csv_path)
        
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            logger.info(f"Sampled {sample_size:,} rows from dataset")
        else:
            logger.info(f"Loaded {len(df):,} rows")
        
        # Check required columns
        required = ["comment_text"] + self.label_cols
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        # Remove rows with missing comment text
        initial_count = len(df)
        df = df.dropna(subset=["comment_text"])
        df = df[df["comment_text"].str.strip() != ""]
        logger.info(f"Removed {initial_count - len(df):,} rows with empty/missing comments")
        
        # Clean text data
        logger.info("Cleaning text data...")
        df["cleaned_text"] = df["comment_text"].apply(self.clean_text)
        
        # Remove rows that became empty after cleaning
        df = df[df["cleaned_text"].str.strip() != ""]
        logger.info(f"Final dataset size: {len(df):,} rows")
        
        return df
    
    def analyze_class_distribution(self, df: pd.DataFrame) -> dict:
        """Analyze class distribution in the dataset"""
        distribution = {}
        for col in self.label_cols:
            count = df[col].sum()
            percentage = (count / len(df)) * 100
            distribution[col] = {
                'count': int(count),
                'percentage': round(percentage, 2)
            }
        
        # Multi-label statistics
        df['label_count'] = df[self.label_cols].sum(axis=1)
        multi_label_stats = df['label_count'].value_counts().sort_index()
        
        distribution['multi_label_stats'] = multi_label_stats.to_dict()
        distribution['total_samples'] = len(df)
        distribution['clean_samples'] = len(df[df['label_count'] == 0])
        distribution['toxic_samples'] = len(df[df['label_count'] > 0])
        
        return distribution
    
    def handle_class_imbalance(self, df: pd.DataFrame, method: str = "undersample") -> pd.DataFrame:
        """Handle class imbalance using various techniques"""
        logger.info(f"Handling class imbalance using {method} method")
        
        if method == "undersample":
            return self._undersample_majority_class(df)
        elif method == "oversample":
            return self._oversample_minority_class(df)
        elif method == "smote":
            return self._apply_smote(df)
        else:
            logger.warning("No class balancing applied")
            return df
    
    def _undersample_majority_class(self, df: pd.DataFrame) -> pd.DataFrame:
        """Undersample majority class (non-toxic comments)"""
        toxic_df = df[df[self.label_cols].sum(axis=1) > 0]
        clean_df = df[df[self.label_cols].sum(axis=1) == 0]
        
        # Sample equal number of clean comments as toxic comments
        target_size = len(toxic_df)
        clean_df_sampled = resample(clean_df, n_samples=target_size, random_state=42)
        
        balanced_df = pd.concat([toxic_df, clean_df_sampled])
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Balanced dataset: {len(balanced_df):,} samples (toxic: {len(toxic_df):,}, clean: {len(clean_df_sampled):,})")
        return balanced_df
    
    def _oversample_minority_class(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oversample minority classes"""
        toxic_df = df[df[self.label_cols].sum(axis=1) > 0]
        clean_df = df[df[self.label_cols].sum(axis=1) == 0]
        
        # Oversample toxic comments to match clean comments
        target_size = len(clean_df)
        toxic_df_oversampled = resample(toxic_df, replace=True, n_samples=target_size, random_state=42)
        
        balanced_df = pd.concat([clean_df, toxic_df_oversampled])
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Balanced dataset: {len(balanced_df):,} samples (toxic: {len(toxic_df_oversampled):,}, clean: {len(clean_df):,})")
        return balanced_df
    
    def _apply_smote(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply SMOTE for synthetic data generation"""
        try:
            from imblearn.over_sampling import SMOTE
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            logger.info("Applying SMOTE for text data...")
            
            # Convert text to TF-IDF features
            vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            X = vectorizer.fit_transform(df['cleaned_text'])
            y = df[self.label_cols].values
            
            # Apply SMOTE
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            # Note: SMOTE creates synthetic features, not text
            # This would require additional processing to convert back to text
            logger.warning("SMOTE applied - synthetic features generated. Text reconstruction needed.")
            
            return df  # Return original for now
            
        except ImportError:
            logger.warning("imbalanced-learn not installed. Skipping SMOTE.")
            return df
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets"""
        logger.info(f"Splitting data: train={1-test_size-val_size:.1%}, val={val_size:.1%}, test={test_size:.1%}")
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42, stratify=df[self.label_cols].sum(axis=1) > 0
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size_adjusted, random_state=42, 
            stratify=train_val_df[self.label_cols].sum(axis=1) > 0
        )
        
        logger.info(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str = "data/processed"):
        """Save processed datasets"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        train_df.to_csv(f"{output_dir}/train_processed.csv", index=False)
        val_df.to_csv(f"{output_dir}/val_processed.csv", index=False)
        test_df.to_csv(f"{output_dir}/test_processed.csv", index=False)
        
        logger.info(f"Saved processed data to {output_dir}")
        
        # Save class distribution statistics
        stats = {
            'train': self.analyze_class_distribution(train_df),
            'val': self.analyze_class_distribution(val_df),
            'test': self.analyze_class_distribution(test_df)
        }
        
        import json
        with open(f"{output_dir}/class_distribution.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("Saved class distribution statistics")

def main():
    """Main preprocessing pipeline"""
    preprocessor = DataPreprocessor()
    
    # Load and clean data
    df = preprocessor.load_and_clean_data("data/train.csv", sample_size=50000)  # Use sample for faster processing
    
    # Analyze original distribution
    original_dist = preprocessor.analyze_class_distribution(df)
    logger.info("Original class distribution:")
    for label, stats in original_dist.items():
        if label != 'multi_label_stats' and isinstance(stats, dict):
            logger.info(f"  {label}: {stats['count']} ({stats['percentage']}%)")
    
    # Handle class imbalance
    balanced_df = preprocessor.handle_class_imbalance(df, method="undersample")
    
    # Analyze balanced distribution
    balanced_dist = preprocessor.analyze_class_distribution(balanced_df)
    logger.info("Balanced class distribution:")
    for label, stats in balanced_dist.items():
        if label != 'multi_label_stats' and isinstance(stats, dict):
            logger.info(f"  {label}: {stats['count']} ({stats['percentage']}%)")
    
    # Split data
    train_df, val_df, test_df = preprocessor.split_data(balanced_df)
    
    # Save processed data
    preprocessor.save_processed_data(train_df, val_df, test_df)
    
    logger.info("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
