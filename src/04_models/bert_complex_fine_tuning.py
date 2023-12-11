# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Fine-Tuning BERT (Complex Model) - Topics 🤖
# MAGIC
# MAGIC This notebook fine-tunes a more complex BERT model to classify news topics
# MAGIC
# MAGIC #### Notebook Properties
# MAGIC * Upstream Notebook: `src.engineering.topic_processor`
# MAGIC * Compute Resources: `61 GB RAM, 1 GPU` (maybe?)
# MAGIC * Last Updated: `Dec 10 2023`
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
from datetime import datetime
import os
from typing import Any, Callable
from loguru import logger
import random

from deltalake import DeltaTable
from tqdm.autonotebook import tqdm
from src.utils.io import FileSystemHandler
from src.utils.functions import all_stopwords

import nltk

from transformers import BertTokenizer, TFBertModel, BertConfig
from tensorflow.keras.layers import Bidirectional, LSTM, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional,Conv1D,GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# COMMAND ----------

# Check for GPU availability
gpus = tf.config.experimental.list_physical_devices('GPU')

# Setup strategy based on the available device: GPU or CPU
if gpus:
    try:
        # If GPUs are available, use MirroredStrategy for distributed training
        tf.config.experimental.set_memory_growth(gpus[0], True)  # Optional: Enable memory growth
        strategy = tf.distribute.MirroredStrategy()
        print("Running on GPU:", gpus[0])
    except RuntimeError as e:
        print(e)
else:
    # If no GPUs are available, use the default strategy that works on CPU and single GPU
    strategy = tf.distribute.get_strategy()
    print("Running on CPU")

print("REPLICAS: ", strategy.num_replicas_in_sync)

# COMMAND ----------

# DBTITLE 1,Settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_info_rows", 10_000_000)

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

topic_to_id = {topic: id for id, topic in enumerate(df["simple_topic"].unique())}
id_to_topic = {id: topic for topic, id in topic_to_id.items()}

df_y["simple_topic_id"] = df_y["simple_topic"].map(topic_to_id)
df_y[["simple_topic", "simple_topic_id"]].sample(5)

# COMMAND ----------

df_y = df_y.dropna(subset=["title_clean"]).dropna(subset=["simple_topic_id"])
print(df_y.shape)

# COMMAND ----------

BASE_DIR_NAME: str = "experiment_results"
NOTEBOOK_DIR_NAME: str = "bert"

# COMMAND ----------

def train_complex_bert_classifier(
    df: pd.DataFrame,
    input_col: str,
    target_col: str,
    sample_size: int | None = None,
    target_col_inverse_mapping: dict | None = None,
    max_len: int = 64,
    bert_model_name: str = "bert-base-uncased",
    learning_rate: float = 0.00001,
    batch_size: int = 32,
    epochs: int = 3,
):
    title_topic_dir_name: str = (
        f"./{BASE_DIR_NAME}"
        + f"/{NOTEBOOK_DIR_NAME}"
        + f"/bert_complex"
        + f"/{datetime.utcnow().strftime('%Y%m%d-%H%M')}"
    )
    os.makedirs(title_topic_dir_name, exist_ok=True)

    sample_size = sample_size if sample_size else len(df)
    sample_df: pd.DataFrame = df.sample(sample_size)

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    def encode_texts(texts):
        return tokenizer.batch_encode_plus(
            texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="tf",
        )

    input_values = encode_texts(sample_df[input_col].values.tolist())
    target_values = sample_df[target_col].values

    input_ids = np.array(input_values["input_ids"])

    X_train, X_test, y_train, y_test = train_test_split(
        input_ids, target_values, test_size=0.2, random_state=42
    )

    bert = TFBertModel.from_pretrained(bert_model_name)

    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    bert_output = bert(input_ids)[0] 

    bi_lstm = Bidirectional(LSTM(128, return_sequences=True))(bert_output)
    avg_pool = GlobalAveragePooling1D()(bi_lstm)
    dropout = Dropout(0.3)(avg_pool)

    intermediate = Dense(128, activation="relu")(dropout)
    intermediate = LayerNormalization()(intermediate)
    intermediate_dropout = Dropout(0.5)(intermediate)

    output = Dense(len(np.unique(target_values)), activation="softmax")(
        intermediate_dropout
    )

    model = Model(inputs=input_ids, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model_history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
    )

    y_pred = np.argmax(model.predict(X_test), axis=1)

    target_classification_report = classification_report(
        y_test,
        y_pred,
        target_names=[
            target_col_inverse_mapping[i]
            for i in sorted(set(y_test) | set(y_pred))
            if i in target_col_inverse_mapping
        ],
    )

    with open(f"{title_topic_dir_name}/classification_report.txt", "w") as f:
        f.write(target_classification_report)

    output_params = dict(
        max_len=max_len,
        bert_model_name=bert_model_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
    )

    with open(f"{title_topic_dir_name}/hyperparameters.json", "w") as f:
        json.dump(output_params, f, indent=4)

# COMMAND ----------

train_complex_bert_classifier(
    df=df_y,
    input_col="title_clean",
    target_col="simple_topic_id",
    target_col_inverse_mapping=id_to_topic,
)

# COMMAND ----------


