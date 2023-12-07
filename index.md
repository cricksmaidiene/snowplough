# Snowplough 🏂

🖋 **Authors**: [Eshwaran Venkat](mailto:eshwaran@ischool@berkeley.edu), under the guidance of [Jennifer Zhu](mailto:zhuxuan@ischool.berkeley.edu)

A suite of descriptive statistics & deep-learning classifiers that analyze media bias in news. These tools evaluate topic and sentiment trends, and consider author and publication factors.

Final project for UC Berkeley [MIDS 266 (Natural Language Processing with Deep Learning)](https://ischoolonline.berkeley.edu/data-science/curriculum/natural-language-processing/). See [Course Repository](https://github.com/datasci-w266)

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

Assigning bias to an article can be subjective, but can dependend broadly on the topic of the article, the publication, the author, and the sentiment of the article. This project aims to build a topic & author classifier, which are in turn used to create descriptive statistics in order to study media bias.

## Data 📇

### AllTheNews

AllTheNews is a popular dataset of news articles that has two versions. Version 1 & 2.

* [Version 2.0](https://components.one/datasets/all-the-news-2-news-articles-dataset) has 2.7 million articles from a number of sources.
* It is a published dataset that is readily downloadable.
* The date range of articles is from January 1, 2016 to April 2, 2020.
* The only metadata available is the article `title`, `publication`, `section`, `author`, `date`, and `content`. We use a subset of these as labels for our classifiers.

## Notebooks 📙

| NB Order Number | Notebook | Section | Description |
| --- | --- | --- | --- |
| 01 | [Ingest Dataset](notebooks/01_ingestion/all_the_news_v2_ingest.html) | Ingestion | Ingests the All The News v2 dataset into a Delta Lake table. |
| 02 | [Exploratory Data Analysis](notebooks/03_analysis/summary_eda.html) | Analysis | Performs exploratory data analysis on the All The News v2 dataset for Summary Statistics. |
| 03 | [Word Counts & Sentiments Processor](notebooks/02_engineering/word_counts_and_sentiments.html) | Engineering | Transformation layer that adds word count fields and sentiment score fields per article |
| 04 | [Sentiment Analysis](notebooks/03_analysis/sentiment_eda.html) | Analysis | Looks at descriptive statistics on sentiment scores across articles, publications and authors to find signals for bias |
| 05 | [News Section Analysis](notebooks/03_analysis/sections_eda.html) | Analysis | Explores newspaper sections for topic-level coalescing and assignment |
| 06 | [Author Analysis](notebooks/03_analysis/author_eda.html) | Analysis | Explores author distributution and slants for simple author classification |
| 07 | [Topic Processor](notebooks/02_engineering/topic_processor.html) | Engineering | Transformation layer that adds topic fields per article using a topic lexicon, and performs additional processing |
| 08 | [LDA Topic Modeling](notebooks/03_analysis/lda_analysis.html) | Modeling | Performs Latent Dirichlet Allocation analysis on the All The News v2 dataset for Topic Modeling |
| 09 | [BERTTopic](notebooks/02_engineering/bert_topic_processor.html) | Engineering | Transformation layer that adds topic fields per article using a BERT-based topic classifier, and performs additional processing |
| 10 | [Standard Classification Models](notebooks/04_models/standard_classification_models.html) | Modeling | Comprehensive set of non-neural network models for Topic &Author classification - Random Forests, Logistic Regression, etc. |
| 10 | [Standard Experiment Results](src/04_models/experiment_results) | Experiments | Classifier Metrics across different trials |
| 11 | [BERT Classifier](notebooks/04_models/bert_classifier.html) | Modeling | BERT-based neural network model for Topic & Author classification |
| 12 | [Bias Index](notebooks/03_analysis/bias_index_and_statistics.html) | Analysis | Calculates bias index and performs preliminary bias analysis using trained classifiers |
