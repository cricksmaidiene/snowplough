# Snowplough 🏂

🖋 **Authors**: [Eshwaran Venkat](mailto:eshwaran@ischool@berkeley.edu), under the guidance of [Jennifer Zhu](mailto:zhuxuan@ischool.berkeley.edu)

Supervised Learning meets Media Analysis: Simple Topic Classification to Explore Bias in News Coverage. Final project for UC Berkeley [MIDS 266 (Natural Language Processing with Deep Learning)](https://ischoolonline.berkeley.edu/data-science/curriculum/natural-language-processing/). See [Course Repository](https://github.com/datasci-w266)

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

Our study introduces an approach to analyze news content, centering on the development of a topic classifier using the extensive `All The News v2` dataset. Our methodology progresses from baseline classifiers to more advanced models, culminating in a fine-tuned BERT classifier, adept at categorizing news articles into distinct topics such as 'Sports,' 'Finance', etc., based on textual features and news metadata.

This classifier is augmented with sentiment analysis and other indicators for a supplemental exploration into media bias, aiming to delineate its various manifestations. The core of our research lies in the robust topic classification, with media bias analysis providing additional insights.
We’ve made the code, notebooks, models and newly generated (topic classified) dataset publicly available. The newly created dataset is listed as All The News v2.1 on Kaggle and a fine-tuned BERT classifier for the same is also made available online.

* **Project Report**: [Download PDF](https://github.com/cricksmaidiene/snowplough/blob/main/snowplough_project_report.pdf)
* **Presentation**: [Download PDF](https://github.com/cricksmaidiene/snowplough/blob/main/presentation.pdf)

---

* **GitHub**: [cricksmaidiene/snowplough](https://github.com/cricksmaidiene/snowplough)
* **Kaggle**: [Coming Soon]()
* **Hugging Face**: [Coming Soon]()

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
| 06 | [Topic Processor](notebooks/02_engineering/topic_processor.html) | Engineering | Transformation layer that adds topic fields per article using a topic lexicon, and performs additional processing |
| 07 | [Topic & Author Analysis](notebooks/03_analysis/topic_and_author_eda.html) | Analysis | Explores the newly labeled and created topics, and how they interact with author distributution and slants |
| 08 | [Standard Classification Models](notebooks/04_models/standard_classification_models.html) | Machine Learning | Comprehensive set of non-neural network models for Topic & Optional Author classification - Random Forests, Logistic Regression, & Naive Bayes |
| 09 | [Neural Network Classifiers](notebooks/04_models/nn_models.html) | Machine Learning | Bi-Directional LSTM and CNN networks are trained for classification of news topics from news titles |
| 10 | [BERT Simple Classifier](notebooks/04_models/bert_simple_model.html) | Machine Learning | A model that minimally fine-tunes a pre-trained BERT Model to classify news topics |
| 11 | [BERT Complex Classifier](notebooks/04_models/bert_complex_model.html) | Machine Learning | A model that adds LSTM and CNN layers on top of a pre-trained BERT model to train the classifier |
| 12 | [Bias Analysis](notebooks/03_analysis/bias_analysis.html) | Analysis | Systematically performs a simple bias analysis on newly labeled topics and sentiments on the news data |
