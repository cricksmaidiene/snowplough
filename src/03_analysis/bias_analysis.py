# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Bias Analysis 🤬
# MAGIC
# MAGIC This notebook conducts a systematic and simple bias analysis using the newly assigned topics from the classifier, together with additional attributes
# MAGIC
# MAGIC #### Notebook Properties
# MAGIC * Upstream Notebook: `src.engineering.topic_processor`
# MAGIC * Compute Resources: `32 GB RAM, 4 CPUs`
# MAGIC * Last Updated: `Dec 10 2023`
# MAGIC
# MAGIC #### Data
# MAGIC
# MAGIC | **Name** | **Type** | **Location Type** | **Description** | **Location** | 
# MAGIC | --- | --- | --- | --- | --- | 
# MAGIC | `all_the_news` | `input` | `Delta` | Read full delta dataset of `AllTheNews` | `catalog/simple_topic/all_the_news.delta` | 

# COMMAND ----------

!python -m spacy download en_core_web_sm -q

# COMMAND ----------

import spacy
nlp = spacy.load("en_core_web_sm")

# COMMAND ----------

# DBTITLE 1,Imports
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import plotly.express as px
from tqdm.autonotebook import tqdm
from deltalake import DeltaTable
from src.utils.io import FileSystemHandler

# COMMAND ----------

# DBTITLE 1,Settings
pd.set_option("display.max_columns", None)
pd.options.plotting.backend = "plotly"
tqdm.pandas()

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
df = df[df.year != 2020]

df = df.dropna(subset=["section"])
"""Drop articles without sections for this analysis"""

print(df.shape)
df.head()

# COMMAND ----------

topic_sentiment_df = (
    df.groupby(["simple_topic", "publication"])
    .agg({"vader_compound_title": "mean"})
    .query("vader_compound_title < -0.25  | vader_compound_title > 0.25")
    .reset_index()
)
topic_sentiment_df.columns = ["topic", "publication", "average_sentiment"]
topic_sentiment_df = topic_sentiment_df.sort_values(by=["topic", "publication"])
topic_sentiment_df

# COMMAND ----------

topic_trend = (
    df.groupby([pd.Grouper(key="date", freq="M"), "simple_topic"])
    .agg({"article": "count", "vader_compound_title": "mean"})
    .reset_index()
)
topic_trend.columns = ["date", "simple_topic", "article_count", "avg_sentiment"]
topic_trend

# COMMAND ----------

pub_topic_trend = (
    df.groupby([pd.Grouper(key="date", freq="M"), "simple_topic", "publication"])
    .agg({"article": "count", "vader_compound_title": "mean"})
    .reset_index()
)
pub_topic_trend.columns = [
    "date",
    "simple_topic",
    "publication",
    "article_count",
    "avg_sentiment",
]
pub_topic_trend

# COMMAND ----------

pub_topic_total = pub_topic_trend.merge(topic_trend, "left", ["date", "simple_topic"])
pub_topic_total.columns = [
    "date",
    "simple_topic",
    "publication",
    "published_articles_in_month",
    "avg_sentiment_of_published_articles",
    "total_articles_in_month",
    "avg_sentiment_of_total_articles",
]
pub_topic_total

# COMMAND ----------

pub_topic_total['published_ratio'] = pub_topic_total['published_articles_in_month'] / pub_topic_total['total_articles_in_month']
pub_topic_total['sentiment_ratio'] = pub_topic_total['avg_sentiment_of_published_articles'] - pub_topic_total['avg_sentiment_of_total_articles']
pub_topic_total

# COMMAND ----------

display(pub_topic_total)

# COMMAND ----------

# Filtering out rows with just 1 article in that month for a topic or publication
df_filtered = pub_topic_total[pub_topic_total["published_articles_in_month"] > 1]

# Calculating the weighted average for published ratio and sentiment ratio
# Weight is the number of articles published by the publication for the topic

# Weighted average published ratio
df_filtered["weighted_published_ratio"] = (
    df_filtered["published_articles_in_month"] * df_filtered["published_ratio"]
)
weighted_avg_published_ratio = (
    df_filtered.groupby(["publication", "simple_topic", "date"])[
        "weighted_published_ratio"
    ].sum()
    / df_filtered.groupby(["publication", "simple_topic", "date"])[
        "published_articles_in_month"
    ].sum()
)

# Weighted average sentiment ratio
df_filtered["weighted_sentiment_ratio"] = (
    df_filtered["published_articles_in_month"] * df_filtered["sentiment_ratio"]
)
weighted_avg_sentiment_ratio = (
    df_filtered.groupby(["publication", "simple_topic", "date"])[
        "weighted_sentiment_ratio"
    ].sum()
    / df_filtered.groupby(["publication", "simple_topic", "date"])[
        "published_articles_in_month"
    ].sum()
)

# Combining the weighted averages to form a dataframe
weighted_avg_df = pd.DataFrame(
    {
        "weighted_avg_published_ratio": weighted_avg_published_ratio,
        "weighted_avg_sentiment_ratio": weighted_avg_sentiment_ratio,
    }
).reset_index()

# Calculating the bias score
# Assuming equal weight for publication ratio and sentiment ratio
weighted_avg_df["deviation_score"] = (
    weighted_avg_df["weighted_avg_published_ratio"]
    + weighted_avg_df["weighted_avg_sentiment_ratio"]
) / 2

weighted_avg_df.head()

# COMMAND ----------

weighted_avg_df.groupby(["publication", pd.Grouper(key="date", freq="M")])[
    ["bias_score"]
].mean().reset_index().plot(
    kind="line", x="date", y="bias_score", color="publication", template="plotly_white"
)

# COMMAND ----------


