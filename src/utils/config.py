"""Simple configuration helpers for the project.

This keeps tests and imports working even if no project-specific
configuration is provided.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def data_dir() -> Path:
	return PROJECT_ROOT / "data"

def models_dir() -> Path:
	return PROJECT_ROOT / "models"

def ensure_dirs():
	(data_dir() / "raw").mkdir(parents=True, exist_ok=True)
	(data_dir() / "processed").mkdir(parents=True, exist_ok=True)
	models_dir().mkdir(parents=True, exist_ok=True)
