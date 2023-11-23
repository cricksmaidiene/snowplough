# Snowplough 🏂

🖋 **Authors**: [Eshwaran Venkat](mailto:eshwaran@ischool@berkeley.edu), under the guidance of [Jennifer Zhu]()

A machine learning model that detects bias in news and media articles.

Final project for UC Berkeley [MIDS 266 (Natural Language Processing)](https://ischoolonline.berkeley.edu/data-science/curriculum/natural-language-processing/). See [Course Repository](https://github.com/datasci-w266)

![](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)
![](https://img.shields.io/badge/Databricks-FF3621.svg?style=for-the-badge&logo=Databricks&logoColor=white)
![](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white)
![](https://img.shields.io/badge/Poetry-60A5FA.svg?style=for-the-badge&logo=Poetry&logoColor=white)
![](https://img.shields.io/badge/Anaconda-44A833.svg?style=for-the-badge&logo=Anaconda&logoColor=white)
![](https://img.shields.io/badge/pandas-150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![](https://img.shields.io/badge/NumPy-013243.svg?style=for-the-badge&logo=NumPy&logoColor=white)
![](https://img.shields.io/badge/TensorFlow-FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![](https://img.shields.io/badge/scikitlearn-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![](https://img.shields.io/badge/Delta-003366.svg?style=for-the-badge&logo=Delta&logoColor=white)
![](https://img.shields.io/badge/Amazon%20S3-569A31.svg?style=for-the-badge&logo=Amazon-S3&logoColor=white)
![](https://img.shields.io/badge/Files-4285F4.svg?style=for-the-badge&logo=Files&logoColor=white)

## About 📰

**Bias** in an article refers to the presence of a certain slant or inclination in the way information is presented. It means that the content is delivered with a partial perspective, favoring one side or viewpoint over others. This can manifest in various forms, such as political bias, cultural bias, or commercial bias, among others. It can be introduced in an article in many ways, such as the choice of words, the tone of the article, the selection of facts, and the presentation of data. It can be introduced intentionally or unintentionally.

## Data 📇

### AllTheNews

AllTheNews is a popular dataset of news articles that has two versions. Version 1 & 2.

* [Version 2.0](https://components.one/datasets/all-the-news-2-news-articles-dataset) has 2.7 million articles from a number of sources.
* It is a published dataset that is readily downloadable.
* The date range of articles is from January 1, 2016 to April 2, 2020.
* This is an unlabelled dataset, and doesn’t have a bias rating per article

## Notebooks 📙

| NB Order Number | Notebook | Section | Description |
| --- | --- | --- | --- |
| 01 | [All The News v2 Ingest](notebooks/01_data_ingestion/all_the_news_v2_ingest.html) | Data Ingestion | Ingests the All The News v2 dataset into a Delta Lake table. |
| 02 | [All The News Summary EDA](notebooks/02_exploratory_data_analysis/all_the_news_summary_eda.html) | Exploratory Data Analysis | Performs exploratory data analysis on the All The News v2 dataset for Summary Statistics. |
| 03 | [All The News Text Preprocessor - Word Counts & Sentiments](notebooks/02_exploratory_data_analysis/preprocessor_wc_and_sentiments.html) | Exploratory Data Analysis | Transformation layer that adds word count fields and sentiment score fields per article |

## Research Design 🔍

### Measuring Bias 🎙

Measuring bias is a subjective task and can be quite challenging. However, there are broadly several approaches:

* **Analyzing Word Choice and Phrasing**: Certain words or phrases may be loaded with connotations that reveal bias.
* **Source and Citation Analysis**: Evaluating the diversity and credibility of the sources and citations used in the article.
* **Comparison with Other Reports**: Comparing the article with other reports on the same topic to identify discrepancies or slants.

Natural Language Processing and Machine Learning models can detect bias by:

* **Keyword and Phrase Analysis**: Identifying and analyzing the usage of biased or subjective words or phrases.
* **Sentiment Analysis**: Evaluating whether the text leans towards a positive or negative sentiment unnecessarily.
* **Statistical Analysis**: Using statistical methods to identify patterns or anomalies that might indicate bias.
* **Machine Learning Models**: Training models to recognize and categorize bias based on historical data and linguistic features.
  
### Bias Indicators 🚨

#### Surface Indicators

* **Strong Adjectives and Adverbs**: Usage of strong or emotional words to describe events, people, or entities.
* **Imbalanced Reporting**: Overemphasis on one viewpoint, while neglecting or undermining others.
* **Lack of Source Diversity**: Reliance on a limited or biased set of sources or experts.
* **Direct Opinions**: Presence of the author’s personal opinions or beliefs

#### Subtle Indicators

* **Framing**: The way information is presented or framed can subtly introduce bias.
* **Choice of Quotes**: Selectively using quotes that support a particular viewpoint.
* **Omission**: Leaving out certain facts, perspectives, or context that might lead to a more balanced understanding.
* **Placement of Information**: Putting biased or opinionated content prominently, while burying neutral or opposing views.
