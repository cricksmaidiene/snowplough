# Snowplough 🏂

A machine learning model that performs topic classification of news articles for media bias analysis. Final project for UC Berkeley MIDS 266 (Natural Language Processing)

Environments:

![](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)
![](https://img.shields.io/badge/Databricks-FF3621.svg?style=for-the-badge&logo=Databricks&logoColor=white)
![](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white)
![](https://img.shields.io/badge/Poetry-60A5FA.svg?style=for-the-badge&logo=Poetry&logoColor=white)

Libraries:

![](https://img.shields.io/badge/Anaconda-44A833.svg?style=for-the-badge&logo=Anaconda&logoColor=white)
![](https://img.shields.io/badge/pandas-150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![](https://img.shields.io/badge/NumPy-013243.svg?style=for-the-badge&logo=NumPy&logoColor=white)
![](https://img.shields.io/badge/TensorFlow-FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![](https://img.shields.io/badge/scikitlearn-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

Data:

![](https://img.shields.io/badge/Delta-003366.svg?style=for-the-badge&logo=Delta&logoColor=white)
![](https://img.shields.io/badge/Amazon%20S3-569A31.svg?style=for-the-badge&logo=Amazon-S3&logoColor=white)
![](https://img.shields.io/badge/Files-4285F4.svg?style=for-the-badge&logo=Files&logoColor=white)

## Installation

Setup anaconda as a virtual environment

```bash
conda create --name snowplough python=3.10 -y
conda activate snowplough
```

Dowload and install snowplough dependencies

```bash
git clone https://github.com/cricksmaidiene/snowplough
cd snowplough
```

Install with poetry:

```bash
poetry install
```

Or with pip:

```bash
pip install .
```

## Tools & Infrastructure

All descriptive analysis, data engineering, processing and baseline modeling was run within Python environment-based Databricks notebooks on CPU backed single-node clusters. Spark was not required, and the main choice for Databricks here was to allow variable sized clusters based on requirements at different project stages. No Databricks-specific commands or dependencies exist, and the **notebooks are agnostic and can be run directly on Jupyter or Google Colab as well**, provided that the Python requirements are met, and the requisite hardware is available. A custom Delta Lake (an open source file format on top of apache parquet) handler to store data locally in the file system or on AWS S3 was used, in order to manage memory better for the size of All The News v2. The neural network based classifiers were trained on P-class and G-class instance-type GPUs made available through AWS & Databricks. Mlflow was used to track and save experimental results for trial and error of hyperparameter tuning

## Data Layer

This project utilizes [Delta Lake](https://delta.io/) for data storage. The storage location is flexible between [AWS S3](https://aws.amazon.com/s3/) or Local Filesystem. The data layer is abstracted away from the user and can be specified when calling `FileSystemHandler` from `src.utils.io` in notebooks.

Example:

```python
from src.utils.io import FileSystemHandler

# AWS S3
datafs = FileSystemHandler("s3", s3_bucket="snowplough-mids")

# Local Filesystem
datafs = FileSystemHandler("local", local_path="/path/to/data/dir")

# List Tables
datafs.listdir("/location/catalog/")
```
