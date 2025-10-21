import logging

logger = logging.getLogger(__name__)

def extract_date_features(df):
    """Extract temporal features from date columns."""
    import pandas as pd
    df = df.copy()

    date_columns = ['start_date', 'end_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df[f'{col}:year'] = df[col].dt.year
            df[f'{col}:month'] = df[col].dt.month
            df[f'{col}:day'] = df[col].dt.day
            df[f'{col}:dayofweek'] = df[col].dt.dayofweek

    # Calculate campaign duration if both dates exist
    if 'start_date' in df.columns and 'end_date' in df.columns:
        df['campaign_duration'] = (df['end_date'] - df['start_date']).dt.days

    return df

def one_hot_encode_categorical(df, categorical_columns: list = None):
    """One-hot encode categorical variables."""
    import pandas as pd
    if categorical_columns is None:
        categorical_columns = ['chain_id', 'format']

    df_encoded = pd.get_dummies(df, columns=categorical_columns, prefix_sep=':')
    return df_encoded

def remove_outliers(df, target_col: str = 'budget', threshold: float = 3):
    """Remove outliers using z-score method."""
    from scipy import stats
    if target_col in df.columns:
        z_scores = stats.zscore(df[target_col])
        df_clean = df[(z_scores < threshold) & (z_scores > -threshold)].copy()
        logger.info(f"Removed {len(df) - len(df_clean)} outliers from {len(df)} samples")
        return df_clean
    return df

def create_additional_features(df):
    """Create additional engineered features."""
    df = df.copy()

    # Area feature if dimensions exist
    if 'height' in df.columns and 'width' in df.columns:
        df['ad_area'] = df['height'] * df['width']

    # You can add more feature engineering here based on your domain knowledge
    return df