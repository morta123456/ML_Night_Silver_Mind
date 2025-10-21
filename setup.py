from setuptools import setup, find_packages

setup(
    name="mlnight-competition",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "mlflow>=1.20.0",
    ],
    python_requires=">=3.8",
)