# Snowplough 🏂

A machine learning model that detects bias in news and media articles. Final project for UC Berkeley MIDS 266 (Natural Language Processing)

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
