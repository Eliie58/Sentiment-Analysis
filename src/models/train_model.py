"""
Module for training Sentiment Analysis Model
"""

import joblib
import logging
import pandas as pd
from typing import Dict, Tuple

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from ..features.build_features import custom_encoder, FeatureEngineer

logging.basicConfig(level=logging.INFO)


def train_model(data: pd.DataFrame,
                parameters: Dict = {'max_features': ['sqrt'],
                                    'n_estimators': [50],
                                    'max_depth': [None],
                                    'min_samples_split': [5],
                                    'min_samples_leaf': [1],
                                    'bootstrap': [True]})\
        -> Tuple[RandomForestClassifier, FeatureEngineer]:
    """
    Function to train RandomForestClassifier on the input data.

    :param data: Input data to be used for training.
    :type data: Pandas DataFrame
    :parm parameters: Params used in GridSearchCV for best classifier.
    :type parameters: Dictionary
    :return: The trained model and fitted feature engineer.
    :rtype: Tuple
    """
    texts = data['text']
    labels = data['label']

    feature_engineer = FeatureEngineer()
    feature_engineer.fit(texts)

    texts = feature_engineer.transform(texts, labels)
    labels = custom_encoder(labels)

    grid_search = GridSearchCV(RandomForestClassifier(
    ), parameters, cv=2, return_train_score=True, n_jobs=-1, verbose=4)
    grid_search.fit(texts, labels)
    logging.info('Best Params : %s', grid_search.best_params_)
    logging.info('Best Score : %s', grid_search.best_score_)

    classifier = grid_search.best_estimator_

    joblib.dump(classifier, 'classifier.pkl')
    joblib.dump(feature_engineer, 'feature_engineer.pkl')

    return classifier, feature_engineer
