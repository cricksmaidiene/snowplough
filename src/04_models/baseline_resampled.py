# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Baseline Resampled 🍡
# MAGIC
# MAGIC This notebook tries baseline models, but after making class distributions uniform
# MAGIC
# MAGIC #### Notebook Properties
# MAGIC * Upstream Notebook: `src.engineering.topic_processor`
# MAGIC * Compute Resources: `61 GB RAM, 1 GPU` 
# MAGIC * Last Updated: `Dec 4 2023`
# MAGIC
# MAGIC #### Data
# MAGIC
# MAGIC | **Name** | **Type** | **Location Type** | **Description** | **Location** | 
# MAGIC | --- | --- | --- | --- | --- | 
# MAGIC | `all_the_news` | `input` | `Delta` | Read full delta dataset of `AllTheNews` | `catalog/simple_topic/all_the_news.delta` | 

# COMMAND ----------

# DBTITLE 1,Imports
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import json
import re
import os
from typing import Any, Callable
from loguru import logger

import plotly.figure_factory as ff

from deltalake import DeltaTable
from tqdm.autonotebook import tqdm
from src.utils.io import FileSystemHandler
from src.utils.functions import all_stopwords

import nltk
import mlflow
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# COMMAND ----------

# DBTITLE 1,Settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_info_rows", 10_000_000)
pd.set_option("display.max_info_columns", 1_000)

pd.options.plotting.backend = "plotly"

tqdm.pandas()
nltk.download("wordnet")
nltk.download('omw-1.4')
datafs = FileSystemHandler("s3")

# COMMAND ----------

# DBTITLE 1,Input Parameters
LIMIT_PARTITIONS: int | None = None
"""An input parameter to limit the number of table partitions to read from delta. Useful to perform EDA on a sample of data."""

SHUFFLE_PARTITIONS: bool = False
"""Whether to randomize the partitions before reading"""

INPUT_TABLE: str = "all_the_news" 
INPUT_CATALOG: str = "simple_topic"

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Read Data

# COMMAND ----------

# DBTITLE 1,Read Input Data
atn_delta_table: DeltaTable = datafs.read_delta(
    table=INPUT_TABLE,
    catalog_name=INPUT_CATALOG,
    as_pandas=False,
)

df: pd.DataFrame = datafs.read_delta_partitions(
    delta_table=atn_delta_table,
    N_partitions=LIMIT_PARTITIONS,
    shuffle_partitions=SHUFFLE_PARTITIONS,
)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(by=["date"])

df = df[["date", "publication", "author", "title", "article", "section", "simple_topic"]]

print(df.shape)
df.head()

# COMMAND ----------

df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Basic Preprocessing
# MAGIC
# MAGIC * Filtering Rows to a Single Year
# MAGIC * Cleaning up stopwords, lemmatization, case normalization and other tweaks to articles and titles

# COMMAND ----------

# df_y = df[df.date.dt.year == 2019]
df_y = df.copy()
print(df_y.shape)

# COMMAND ----------

lemmatizer = nltk.stem.WordNetLemmatizer()


def preprocess_text(x: str) -> str:
    x = x.lower()
    x = re.sub(r"\W", " ", x)
    x = re.sub(r"\s+", " ", x)
    y = x.split()
    y = [word for word in y if word not in all_stopwords]
    y = [lemmatizer.lemmatize(word) for word in y]
    return " ".join(y)

# COMMAND ----------

df_y["title_clean"] = df_y["title"].dropna().progress_apply(preprocess_text)
df_y[["title", "title_clean"]].sample(5)

# COMMAND ----------

df_y["article_clean"] = df_y["article"].dropna().progress_apply(preprocess_text)
df_y[["article", "article_clean"]].sample(5)

# COMMAND ----------

sample_df = df_y.copy()
sample_df = sample_df.dropna(subset=["title_clean"]).replace([np.nan], [None])

article_count_threshold = int(sample_df.simple_topic.value_counts().max() * 0.05)
print(f"{article_count_threshold=}")

# Filter out classes with fewer samples than the threshold
value_counts = sample_df["simple_topic"].value_counts()
to_remove = value_counts[value_counts <= article_count_threshold].index
sample_df = sample_df[~sample_df["simple_topic"].isin(to_remove)]

# Resample
df_resampled = pd.DataFrame()
for category in sample_df["simple_topic"].unique():
    df_category = sample_df[sample_df["simple_topic"] == category]
    df_resampled = pd.concat(
        [df_resampled, resample(df_category, n_samples=article_count_threshold)]
    )

# Shuffle the resampled dataframe
df_resampled = df_resampled.sample(frac=1).reset_index(drop=True)
sample_df = df_resampled.copy()

print(sample_df.shape)

# COMMAND ----------

sample_df.simple_topic.value_counts()

# COMMAND ----------

BASE_DIR_NAME: str = "experiment_results"
NOTEBOOK_DIR_NAME: str = "baseline_resampled"

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Declare Training Functions

# COMMAND ----------

baseline_model_callables: dict[str, Callable] = dict(
    logistic_regression=LogisticRegression,
    naive_bayes=MultinomialNB,
    random_forest=RandomForestClassifier,
)

# COMMAND ----------

def train_generic_model(model_classifier: Any, X_train, y_train, X_test,) -> Any:
    model_classifier.fit(X_train, y_train)
    model_predictions = model_classifier.predict(X_test)
    return model_predictions


def get_trained_model_stats(
    y_test, model_predictions
) -> tuple[float, float, float, float]:
    accuracy = accuracy_score(y_test, model_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, model_predictions, average="weighted"
    )
    return accuracy, precision, recall, f1


def get_train_and_test_data(
    vectorizer,
    dataframe: pd.DataFrame,
    input_col: str,
    target_col: str,
    test_frac: float = 0.2,
):
    input_features = vectorizer.fit_transform(dataframe[input_col])
    X_train, X_test, y_train, y_test = train_test_split(
        input_features, dataframe[target_col], test_size=test_frac, random_state=50
    )
    return X_train, X_test, y_train, y_test

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Title -> Topic Classifiers

# COMMAND ----------

TITLE_TOPIC_TF_IDF_MAX_FEAT: int = 5_000
TITLE_TOPIC_TFIDF_EXP_NAME: str = "title_topic_tfidf_5p"

title_topic_tfidf_baseline_results: dict = {}
title_topic_tfidf_vectorizer = TfidfVectorizer(
    max_features=TITLE_TOPIC_TF_IDF_MAX_FEAT,
)

title_topic_tfidf_dir_name: str = (
    f"./{BASE_DIR_NAME}"
    + f"/{NOTEBOOK_DIR_NAME}"
    + f"/{TITLE_TOPIC_TFIDF_EXP_NAME}"
    + f"_{TITLE_TOPIC_TF_IDF_MAX_FEAT}"
)
os.makedirs(title_topic_tfidf_dir_name, exist_ok=True)

# COMMAND ----------

(
    X_title_topic_tfidf_train,
    X_title_topic_tfidf_test,
    y_title_topic_tfidf_train,
    y_title_topic_tfidf_test,
) = get_train_and_test_data(
    title_topic_tfidf_vectorizer,
    sample_df,
    "title_clean",
    "simple_topic",
)

for model_name, model_callable in baseline_model_callables.items():
    logger.info(f"Training: {model_name}")
    classifier_instance = model_callable()
    classifier_predictions = train_generic_model(
        model_classifier=classifier_instance,
        X_train=X_title_topic_tfidf_train,
        y_train=y_title_topic_tfidf_train,
        X_test=X_title_topic_tfidf_test,
    )
    title_topic_tfidf_baseline_results[model_name] = classifier_predictions

    model_str: str = classification_report(
        y_title_topic_tfidf_test, classifier_predictions
    )
    with open(f"{title_topic_tfidf_dir_name}/{model_name}.txt", "w") as f:
        f.write(model_str)

    logger.info(f"Complete: {model_name}")

# COMMAND ----------


