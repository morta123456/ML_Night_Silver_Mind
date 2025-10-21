import pytest

# Skip test if pandas is not installed
pd = pytest.importorskip('pandas')
from src.models.train import train_model


def test_train_model_signature():
	# create tiny dataset
	X = pd.DataFrame({'f1': [1, 2, 3, 4], 'f2': [2, 3, 4, 5]})
	y = pd.Series([10, 20, 30, 40])

	model = train_model(X, y, params={'n_estimators': 1, 'max_depth': 1, 'learning_rate': 0.5, 'random_state': 42}, use_mlflow=False)
	assert hasattr(model, 'predict')
