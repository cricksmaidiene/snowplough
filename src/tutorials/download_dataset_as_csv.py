# Databricks notebook source
import warnings
warnings.filterwarnings("ignore")

from deltalake import DeltaTable
import pandas as pd
from tqdm.autonotebook import tqdm
from src.utils.io import FileSystemHandler

# COMMAND ----------

LIMIT_PARTITIONS: int | None = None
"""An input parameter to limit the number of table partitions to read from delta. Useful to perform EDA on a sample of data."""

SHUFFLE_PARTITIONS: bool = False
"""Whether to randomize the partitions before reading"""

INPUT_TABLE: str = "all_the_news" 
INPUT_CATALOG: str = "simple_topic"

datafs = FileSystemHandler("s3")

# COMMAND ----------

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

df = df[~df.is_geo]
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(by=["date"])

df = (
    df.dropna(subset=["title"])
    .dropna(subset=["article"])
    .dropna(subset=["simple_topic"])
)

df = df[
    ["date", "publication", "author", "title", "section", "simple_topic"]
]
df = df[df.simple_topic != "Commercial Business"]
# df = df[df.date.dt.year == 2019]

print(df.shape)
df.head()

# COMMAND ----------

display(df)

# COMMAND ----------


