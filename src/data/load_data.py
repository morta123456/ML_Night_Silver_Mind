import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_datasets(train_path: str, test_path: str, submission_path: str = None) -> tuple:
    """Load and preprocess raw datasets."""
    try:
        train = pd.read_csv(train_path).set_index('campaign_id')
        test = pd.read_csv(test_path).set_index('campaign_id')
        
        if submission_path:
            submission = pd.read_csv(submission_path)
        else:
            submission = None
            
        logger.info(f"Train shape: {train.shape}, Test shape: {test.shape}")
        return train, test, submission
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise