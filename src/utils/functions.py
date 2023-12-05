"""Helper functions to preprocess and transform data."""

import inflection
from gensim.utils import simple_preprocess
from collections import Counter
import nltk
import pandas as pd

nltk.download("stopwords")

stop_words: list[str] = list(set(nltk.corpus.stopwords.words("english")))
all_stopwords: list[str] = ["news"] + stop_words


def get_topics_for_section(
    x: str, simple_topic_lexicon: dict[str, list[str]]
) -> list[str] | None:
    """Looks up a topic lexicon dictionary to map newspaper sections to news topics."""
    topic_list: list[str] = []
    x_list: list[str] = [
        inflection.singularize(split_h)
        for x_str in simple_preprocess(x)
        for split_h in x_str.split("-")
    ]
    x_str: str = inflection.singularize(" ".join(x_list))

    for topic_name, search_list in simple_topic_lexicon.items():
        for search_string in search_list:

            search_strings_to_compare = []
            add_topic: bool = False

            if len(search_string) < 5:
                # account for small words as being separate from larger strings (ex. car, bus, war, etc.)
                search_string_suffix = search_string + " "
                search_string_prefix = " " + search_string
                search_strings_to_compare += [
                    search_string_prefix,
                    search_string_suffix,
                ]
                # check if direct comparison is first met
                add_topic = search_string == x_str

                if not add_topic:
                    add_topic = any([search_string == tok for tok in x_list])

                # check if each word of input string starts or ends with prefix and suffix processed search string
                if not add_topic:
                    add_topic = any(
                        [
                            x_str.startswith(s_str) or x_str.endswith(s_str)
                            for x_str in x_list
                            for s_str in search_strings_to_compare
                        ]
                    )

            else:
                search_strings_to_compare = [search_string]
                # check if input string is equal to or starts with the search string
                add_topic = (search_string == x_str) or (
                    x_str.startswith(search_string)
                )

                # check if each word of input string is equal to or starts with the search string
                if not add_topic:
                    add_topic = any(
                        [
                            (search_string == x_str)
                            or (x_str.startswith(search_string))
                            for x_str in x_list
                        ]
                    )

            if add_topic:
                topic_list.append(topic_name)

    return topic_list if topic_list else None


def assign_simple_topics_to_dataframe(
    article_level_dataframe: pd.DataFrame,
    simple_topic_lexicon: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Takes in a dataframe of articles, maps sections to topics and returns the section level dataframe to re-map."""
    df: pd.DataFrame = article_level_dataframe.copy()
    section_df = df["section"].value_counts().to_frame().reset_index()
    section_df.columns = ["section", "article_count"]

    section_df["section_clean"] = (
        section_df["section"]
        .drop_duplicates()
        .str.lower()
        .apply(simple_preprocess)
        .apply(lambda cell: " ".join([c for c in cell if c not in all_stopwords]))
    )
    section_df = section_df.replace([""], [None])

    section_df["simple_section_topics"] = (
        section_df["section_clean"]
        .dropna()
        .apply(
            lambda cell: get_topics_for_section(
                cell, simple_topic_lexicon=simple_topic_lexicon
            )
        )
    )

    section_df["simple_topic"] = (
        section_df["simple_section_topics"]
        .dropna()
        .apply(lambda cell: Counter(cell).most_common(1)[0][0])
    )

    simple_topic_mapping: dict[str, str] = (
        section_df[(~section_df.simple_topic.isna())]
        .set_index("section")[["simple_topic"]]
        .to_dict()["simple_topic"]
    )

    df["simple_topic"] = df["section"].map(simple_topic_mapping)

    return df, section_df