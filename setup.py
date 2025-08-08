from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="smartfeatureselectml",
    version="0.1.6",
    author="Shreenidhi T H",
    author_email="shreenidhi1903@example.com",
    description="A Python package for automated feature selection in ML workflows.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SHREENIDHI1903/AutoFeatureSelect",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "shap",
        "statsmodels"

    ],
)
