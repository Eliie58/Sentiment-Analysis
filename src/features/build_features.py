"""
Module for Building features
"""
import re
from typing import List, Tuple

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

lm = WordNetLemmatizer()

nltk.download('stopwords')
nltk.download('wordnet')


def custom_encoder(labels: pd.Series) -> pd.Series:
    """
    This function encodes labels into positive
    or negative sentiment.

    :param labels: The labels to to encoded.
    :type labels: pandas Series
    :return: The encoded labels
    :rtype: pandas Series
    """
    labels = labels.replace(to_replace="surprise", value=1)
    labels = labels.replace(to_replace="love", value=1)
    labels = labels.replace(to_replace="joy", value=1)
    labels = labels.replace(to_replace="fear", value=0)
    labels = labels.replace(to_replace="anger", value=0)
    labels = labels.replace(to_replace="sadness", value=0)
    return labels


def text_transformation(texts: pd.Series) -> List:
    """
    This function transforms cleans input text by removing stop
    words and applying lemmatization.

    :param texts: The texts to be transformed.
    :type texts: Pandas Series.
    :return: The cleaned transformed texts.
    :rtype: Python List
    """
    corpus = []
    for item in texts:
        new_item = re.sub('[^a-zA-Z]', ' ', str(item))
        new_item = new_item.lower()
        new_item = new_item.split()
        new_item = [lm.lemmatize(word) for word in new_item if word not in set(
            stopwords.words('english'))]
        corpus.append(' '.join(str(x) for x in new_item))
    return corpus


class FeatureEngineer:
    """
    Class for feature engineering.
    """

    def __init__(self):
        self.count_vectorizer = CountVectorizer(ngram_range=(1, 2))

    def fit(self, texts: pd.Series) -> None:
        """
        Fit the Count Vectorizer on the input texts.

        :param texts: Input texts used for fitting.
        :type texts: Pandas series.
        """
        corpus = text_transformation(texts)
        self.count_vectorizer.fit(corpus)

    def transform(self, texts: pd.Series,
                  labels: pd.Series = None) -> Tuple[pd.Series, List]:
        """
        Apply transformation on the input texts and labels.

        :param texts: Input texts for transformation.
        :type texts: Pandas series.
        :param labels: Input labels for transformation.
        :type labels: Pandas series.
        :return: The transformed texts and labels
        :rtype: Tuple
        """
        corpus = text_transformation(texts)
        texts = self.count_vectorizer.transform(corpus)

        if labels is not None:
            labels = custom_encoder(labels)

        return texts, labels
