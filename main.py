import logging
import argparse
from pathlib import Path

from src.data.load_data import load_datasets
from src.features.engineer_features import (
    extract_date_features, 
    one_hot_encode_categorical, 
    remove_outliers,
    create_additional_features
)
from src.models.train import train_model, cross_validate_model
from src.models.predict import generate_submission, save_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='ML Night Competition Pipeline')
    parser.add_argument('--train_path', default='data/raw/train.csv', help='Path to training data')
    parser.add_argument('--test_path', default='data/raw/test.csv', help='Path to test data')
    parser.add_argument('--submission_path', default='data/raw/sample_submission.csv', help='Path to submission template')
    parser.add_argument('--output_path', default='data/processed/submission.csv', help='Output path for submission')
    
    args = parser.parse_args()
    
    try:
        # Load data
        logger.info("Loading datasets...")
        train, test, submission = load_datasets(args.train_path, args.test_path, args.submission_path)
        
        # Feature engineering
        logger.info("Engineering features...")
        train = extract_date_features(train)
        test = extract_date_features(test)
        
        train = create_additional_features(train)
        test = create_additional_features(test)
        
        train = one_hot_encode_categorical(train)
        test = one_hot_encode_categorical(test)
        
        # Remove outliers from training data only
        train = remove_outliers(train)
        
        # Prepare features and target
        X = train.drop('budget', axis=1)
        y = train['budget']
        
        # Ensure test data has same columns as training data
        test = test.reindex(columns=X.columns, fill_value=0)
        
        # Train model
        logger.info("Training model...")
        model = train_model(X, y)
        
        # Generate submission
        logger.info("Generating submission...")
        generate_submission(model, test, submission, args.output_path)
        
        # Save model
        save_model(model)
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()