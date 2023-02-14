"""
Module for Sentiment prediction in sentences.
"""
import joblib
from typing import List

from numpy.typing import ArrayLike
from sklearn.ensemble import RandomForestClassifier

from ..features.build_features import FeatureEngineer


def predict(texts: List,
            model: RandomForestClassifier = None,
            feature_engineer: FeatureEngineer = None) -> ArrayLike:
    """
    Predict sentiment of input texts.

    :param texts: Input texts.
    :type texts: List
    :param model: Model to use for sentiment prediction.
    :type model: RandomForestClassifier.
    :param feature_engineer: FeatureEngineer to use for pre processing.
    :type feature_engineer: FeatureEngineer.
    :return: Sentiment Predictions.
    :rtype: Numpy array.
    """
    if model is None:
        model = joblib.load('classifier.pkl')

    if feature_engineer is None:
        feature_engineer = joblib.load('feature_engineer.pkl')

    texts, _ = feature_engineer.transform(texts)

    predictions = model.predict(texts)

    return predictions
