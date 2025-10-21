import pytest

# Skip if pandas not available
pd = pytest.importorskip('pandas')
from src.data.load_data import load_datasets
from src.utils.config import ensure_dirs, data_dir


def test_ensure_dirs_creates_paths(tmp_path, monkeypatch):
	# Point project root to a temp dir
	monkeypatch.setattr('src.utils.config.PROJECT_ROOT', tmp_path)
	ensure_dirs()
	assert (tmp_path / 'data' / 'raw').exists()
	assert (tmp_path / 'data' / 'processed').exists()


def test_load_datasets_handles_missing_file(tmp_path):
	# create empty CSVs
	train = tmp_path / 'train.csv'
	test = tmp_path / 'test.csv'
	train.write_text('campaign_id,budget\n')
	test.write_text('campaign_id\n')

	with pytest.raises(Exception):
		# since index column campaign_id is missing values, loading should still raise or return tuple
		load_datasets(str(train), str(test))
