import pytest

# Skip tests if pandas or scipy are not installed in the environment where tests are run
pd = pytest.importorskip('pandas')
np = pytest.importorskip('numpy')
from src.features.engineer_features import extract_date_features, remove_outliers

def test_extract_date_features():
    """Test date feature extraction."""
    df = pd.DataFrame({
        'start_date': ['2020-01-01', '2020-02-15'],
        'end_date': ['2020-01-05', '2020-02-20']
    })
    
    result = extract_date_features(df)
    
    assert 'start_date:year' in result.columns
    assert 'end_date:month' in result.columns
    assert 'campaign_duration' in result.columns
    assert result['campaign_duration'].iloc[0] == 4

def test_remove_outliers():
    """Test outlier removal."""
    df = pd.DataFrame({
        'budget': [100, 200, 300, 1000, 1500]  # Last two are outliers
    })
    
    result = remove_outliers(df, threshold=2)
    
    assert len(result) == 3  # Should remove 2 outliers

def test_feature_engineering_integration():
    """Test the entire feature engineering pipeline."""
    df = pd.DataFrame({
        'start_date': ['2020-01-01'],
        'end_date': ['2020-01-05'],
        'chain_id': ['A'],
        'format': ['B'],
        'budget': [100]
    })
    
    # Test that pipeline runs without errors
    from src.features.engineer_features import (
        extract_date_features, 
        one_hot_encode_categorical,
        create_additional_features
    )
    
    df = extract_date_features(df)
    df = create_additional_features(df)
    df = one_hot_encode_categorical(df)
    
    assert 'chain_id:A' in df.columns
    assert 'format:B' in df.columns