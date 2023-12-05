# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # News Sections EDA 🎙
# MAGIC
# MAGIC This is an exploration of topic labels through the section names of the dataset
# MAGIC
# MAGIC #### Notebook Properties
# MAGIC * Upstream Notebook: `src.engineering.word_counts_and_sentiments`
# MAGIC * Compute Resources: `32 GB RAM, 4 CPUs` (when not performing EDA on a sample of data)
# MAGIC * Last Updated: `Nov 29 2023`
# MAGIC
# MAGIC #### Data
# MAGIC
# MAGIC | **Name** | **Type** | **Location Type** | **Description** | **Location** | 
# MAGIC | --- | --- | --- | --- | --- | 
# MAGIC | `all_the_news` | `input` | `Delta` | Read full delta dataset of `AllTheNews` | `catalog/text_eda/all_the_news.delta` | 

# COMMAND ----------

# DBTITLE 1,Imports
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
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
INPUT_CATALOG: str = "text_eda"

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

df = df.dropna(subset=['section'])
"""Drop articles without sections for this analysis"""

print(df.shape)
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Section Analysis
# MAGIC
# MAGIC ### Variation in Sections
# MAGIC * The section of an article highlights which section of the newspaper it appears in
# MAGIC * This can be highly subjective since **different publications may have intersecting but difference section names for the same topic** (ex. `Tech by VICE`, `Technology News`)
# MAGIC
# MAGIC ### Missing Sections
# MAGIC Moreover, not all articles have an assigned section. 
# MAGIC * For our initial analysis, we will ignore articles without a section, as we cannot model topics easily for these. 
# MAGIC * We may still consider an **unsupervised clustering of these articles to see if they can be imputed for topics**.

# COMMAND ----------

# DBTITLE 1,Articles with Sections
df["section"].dropna().shape

# COMMAND ----------

# DBTITLE 1,Unique Raw Section Names
df["section"].dropna().drop_duplicates().shape

# COMMAND ----------

# DBTITLE 1,Article Count by Section
df["section"].value_counts().head(50).plot(
    kind="bar", template="plotly_white", title="Article Count by Section"
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Analysis of Section Summary
# MAGIC
# MAGIC * `World News` is too generic to be useful. These articles will have to be reduced to further sections or not considered for the study
# MAGIC * Sections that deal with specific countries or cities should be assigned to a separate column indicating geography
# MAGIC   * These also require coalescing (ex. `us` vs. `U.S.`)
# MAGIC * Sections across publications should be coalesced (ex. `Tech by VICE` and `Technology News`)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Preprocess Section Names
# MAGIC
# MAGIC Remove obviously bad section names and solve for the low hanging fruit. 
# MAGIC
# MAGIC * Remove sections that belong to only one article, in total, and only one section per article per year.
# MAGIC * Normalize all sections to Title Case

# COMMAND ----------

# DBTITLE 1,Imposter Sections
df["section"].value_counts().tail(50)

# COMMAND ----------

# DBTITLE 1,Preview: Erroneous Section
df[df.section == 'brilliant-wrong-approximations-of-pi']

# COMMAND ----------

# DBTITLE 1,Is Hyphen Enough to Remove Erroneous Sections?
df[df.section.str.contains("-").fillna(False)].section.value_counts()

# COMMAND ----------

# DBTITLE 1,Assign Opinions to Own Section
df.loc[
    df.section.fillna("").str.lower().apply(lambda cell: cell.startswith("opinion")),
    "section",
] = "Opinion"

# COMMAND ----------

# DBTITLE 1,Nullify Sections that have 1 article only
unique_section_counts: pd.Series = df["section"].value_counts()
sections_with_single_article: list[str] = unique_section_counts[
    unique_section_counts == 1
].index

print(len(sections_with_single_article))

df.loc[df.section.isin(sections_with_single_article), "section"] = None

# COMMAND ----------

# DBTITLE 1,Nullify Sections that have 1 article within each Year
for year in df.year.unique():
    year_unique_section_counts = df[df.year == year]["section"].value_counts()
    sections_with_single_article_in_year: list[str] = year_unique_section_counts[
        year_unique_section_counts == 1
    ].index
    print(
        f"Year {year} has {len(sections_with_single_article_in_year)} single article sections, that will be nullified."
    )
    df.loc[
        (df.section.isin(sections_with_single_article_in_year)) & (df.year == year),
        "section",
    ] = None

# COMMAND ----------

# DBTITLE 1,Result: Low number of spurious section names, Nullified more than 70% of initial distinct sections
df[df.section.str.contains("-").fillna(False)].section.value_counts().tail(50)

# COMMAND ----------

df.section.value_counts().tail(50)

# COMMAND ----------

# DBTITLE 1,Normalize Case for Sections
df['section'] = df['section'].dropna().str.title()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Temporal Section Analysis

# COMMAND ----------

# DBTITLE 1,Article Counts with Sections over Time
df.groupby(pd.Grouper(key="date", freq="M"))["section"].count().plot(
    kind="line",
    template="plotly_white",
    title="Article Counts with Sections over Time",
)

# COMMAND ----------

# DBTITLE 1,Distinct Sections per Month over Time
df.groupby(pd.Grouper(key="date", freq="M"))["section"].nunique().plot(
    kind="line",
    template="plotly_white",
    title="Distinct Sections per Month over Time",
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Mid-2018 Spike Analysis
# MAGIC
# MAGIC The spike in distinct articles in the middle months of 2018 is interesting, since it bulges off quite significantly from the rest of the distribution. We analyse what extra sections these months contain that do not exist in the rest of 2018 and see how much of a differing factor these introduce.

# COMMAND ----------

# DBTITLE 1,Explore Spike in Distinct Sections in Mid 2018
mid_2018_section_counts: pd.Series = df[
    (df.year == 2018) & (df.month.isin([5, 6, 7, 8]))
].section.value_counts()
mid_2018_section_counts

# COMMAND ----------

# DBTITLE 1,All Sections in the Year
full_year_section_counts: pd.Series = df[
    (df.year == 2018) & (~df.month.isin([5, 6, 7, 8]))
].section.value_counts()
full_year_section_counts

# COMMAND ----------

# DBTITLE 1,Sections that appear in the spike, but not the rest of the Year
mid_2018_section_counts[
    ~mid_2018_section_counts.index.isin(full_year_section_counts.index)
]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC These sections could be appearing as part of a larger hierarchy in other months

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Section Distribution
# MAGIC
# MAGIC > This is purely for studying the training data and not meant as a generalizable solution
# MAGIC
# MAGIC * 57% of articles have sections that appear in more than one publication
# MAGIC * 43% of articles have sections that appear only in that publication
# MAGIC * `~200` sections appear across publications, and `~1500` sections are standalone within a pub

# COMMAND ----------

df = df.dropna(subset=['section'])
print(df.shape)

# COMMAND ----------

# DBTITLE 1,Sections x Publications Summary
section_to_publication_counts: pd.Series = (
    df.groupby(["section"])["publication"].nunique().sort_values(ascending=False)
)
cross_pub_sections: pd.Series = section_to_publication_counts[
    section_to_publication_counts > 1
]
single_pub_sections: pd.Series = section_to_publication_counts[
    section_to_publication_counts == 1
]

print(
    f"Distinct sections that appear across publications: {cross_pub_sections.shape[0]}"
)
print(
    f"Distinct sections that appear in single publications only: {single_pub_sections.shape[0]}"
)

df_cross_sec = df[df.section.isin(cross_pub_sections.index)]
df_single_sec = df[df.section.isin(single_pub_sections.index)]

single_sec_articles = df_single_sec.shape[0] / df.shape[0]
cross_sec_articles = df_cross_sec.shape[0] / df.shape[0]

print(
    f"Share of all Articles with Publication-Only Sections: {single_sec_articles:.2f}"
)
print(
    f"Share of all Articles with Cross-Publication Sections: {cross_sec_articles:.2f}"
)

# COMMAND ----------

df_cross_sec.groupby(
    [
        "section",
        "publication",
    ]
)["article"].count().head(50)

# COMMAND ----------


